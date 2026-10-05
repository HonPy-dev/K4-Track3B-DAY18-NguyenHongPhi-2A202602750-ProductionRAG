# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** HongPhi  
**Khóa:** K4 - Track 3B  
**Ngày hoàn thành:** 2026-10-05

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Với threshold 0.85, semantic tạo 208 chunks (avg 99 ký tự) so với basic 51 chunks (avg 410 ký tự) trên corpus 26 docs — cắt nhỏ hơn 4x để giữ nguyên 1 ý/câu, nhưng min_len = 6 cho thấy câu quá ngắn (tiêu đề bảng) vẫn bị tách riêng, dễ mất ngữ cảnh nếu đứng một mình. Production pipeline dùng hierarchical (104 children từ 11 parents) thay vì semantic vì retrieve child → return parent. |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | RRF giải quyết vấn đề điểm BM25 (TF-IDF, unbounded) và cosine dense (0→1) không so sánh trực tiếp được: chỉ dùng **thứ hạng** (1/(60 + rank + 1)) thay vì điểm thô. Doc xuất hiện ở cả 2 list được cộng dồn → push lên đầu. Lưu ý thực chiến: underthesea nối từ ghép bằng `_` (`nghỉ_phép`), phải `replace("_", " ")` ở **cả corpus lẫn query** thì BM25 mới match được. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | bge-reranker-v2-m3 (cross-encoder) chấm điểm từng cặp (query, doc) nên chính xác hơn bi-encoder dùng cho retrieval — đổi lại O(n) forward passes: benchmark thực tế **~4,547ms/query cho 20 docs trên CPU** (chiếm ~36% thời gian mỗi query), so với ~150ms của hybrid search. Đánh đổi xứng đáng: context_precision 0.925 → 0.992. Chiến lược đúng: retrieve rẻ với top-20, rerank đắt chỉ trên candidates → trả top-3. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Kết quả thực tế: faithfulness 0.808→**0.917**, answer_relevancy 0.695→**0.757**, context_precision 0.925→**0.992**, context_recall 0.900→**0.933** — production thắng cả 4. Bài học lớn: failure analysis cho thấy bottleneck đã chuyển sang **generation** (2/5 failure là faithfulness do LLM tự tính toán ngoài context), retrieval gần như hoàn hảo — đúng tinh thần Diagnostic Tree: đọc metric để biết sửa khâu nào. |
| Contextual embeddings | M5 | `contextual_prepend()` / `_enrich_single_call()` | Chunk hierarchical 256 ký tự thường mất bối cảnh. Combined mode gộp 4 techniques vào 1 call/chunk (104 calls, 331s ≈ 3.2s/chunk) — chi phí API giảm 4x so với 4 hàm riêng. Quan trọng hơn cả enrichment: **retrieve child → return parent** khi sinh context cho LLM giúp faithfulness +0.11 — child nhỏ để tìm chính xác, parent lớn để trả lời đủ evidence. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `ModuleNotFoundError: No module named 'torch'` — dù `importlib.metadata` báo `torch==2.12.0` đang cài. Nguyên nhân: site-packages toàn cục chỉ còn `torch-2.12.0.dist-info` + `functorch`/`torchgen`, **thiếu chính thư mục `torch/`** (cài dở hoặc bị dọn) → pip tưởng torch đã cài nên bỏ qua, import thì chết.
  - `UserWarning: Not enough free disk space to download the file. The expected file size is: 2271.15 MB... only has 1651.56 MB free` — ổ C: đầy 100% (863MB trống) khi tải bge-m3.
  - `ERROR: Could not find a version that satisfies the requirement flit_core<4,>=3.11` khi cài torch từ `--index-url https://download.pytorch.org/whl/cpu` — index PyTorch không chứa flit_core, và pip 22.3 cũ buộc build sdist của typing-extensions.
  - `TypeError: Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'` — cache bge-m3 hỏng do download gián đoạn: snapshot thiếu cả thư mục `1_Pooling/`.
  - `OSError: The paging file is too small for this operation to complete. (os error 1455)` + segfault khi load đồng thời bge-m3 + bge-reranker-v2-m3 (2 × 2.2GB) trên máy 16GB RAM đang chạy nhiều app.
  - `AuthenticationError: 401 - Incorrect API key provided: sk-matkh...` — biến môi trường chứa key placeholder, không phải key thật.
- **Nguyên nhân gốc rễ & Cách debug:**
  - torch hỏng: so sánh output của `importlib.metadata` (chỉ đọc dist-info) với `import torch` (cần thư mục thật) → phát hiện metadata tồn tại nhưng package không. Fix: tạo venv `--system-site-packages` rồi `pip install torch==2.7.1+cpu --ignore-installed --index-url .../cpu --extra-index-url https://pypi.org/simple` để ép cài đè vào venv, đồng thời nâng pip để resolve dependency từ PyPI.
  - Hết đĩa: chuyển HF cache sang ổ D: (`setx HF_HOME D:\hf_cache` + di chuyển các model đã tải) — đồng thời tránh xung đột với dữ liệu khác trên C:.
  - Cache hỏng: liệt kê nội dung snapshot thấy thiếu file → xóa model dir rồi `snapshot_download()` lại toàn vẹn.
  - Commit memory (1455/segfault): đo `FreeVirtualMemory` trước/sau từng thao tác, phân biệt 2 đường load — safetensors `safe_open` (mapping COW, chiếm commit ≈ kích thước file) luôn fail khi bge-m3 đã resident, còn `pytorch_model.bin` qua `torch.load(mmap=True)` (read-only) luôn ổn định. Fix 3 lớp: (1) **class-level shared model cache** cho `DenseSearch`/`CrossEncoderReranker` — nhiều instance dùng chung 1 bản load thay vì 5 bản trong pytest, (2) convert reranker sang `.bin`, (3) giải phóng 2 model retrieval trước khi RAGAS nạp bản embeddings riêng.
  - API key: `OPENAI_API_KEY` trong env là placeholder → dùng key OpenRouter có sẵn trong `.env` của lab trước + `OPENAI_BASE_URL`; đưa `LLM_MODEL` và `resolve_model_dir()` vào config để không hardcode; RAGAS trỏ sang bge-m3 local làm embeddings vì gateway không phục vụ embeddings API.
  - API Qdrant: bản `qdrant-client >= 2.0` đổi `search()` → `query_points()`; đọc changelog và dùng đúng API mới.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Cơ chế cache của huggingface_hub (hub layout `models--org--name/snapshots/blobs`, resume download) → đọc docs HF, hiểu biến `HF_HOME`/`HF_HUB_CACHE`.
  - Windows commit memory khác physical RAM; khác biệt mapping read-only vs copy-on-write giữa `torch.load` và `safetensors` → phân tích bằng thực nghiệm (đo commit trước/sau từng load, thử từng đường load một).
  - Khác biệt bi-encoder (retrieve nhanh, chính xác vừa) vs cross-encoder (chính xác cao, O(n) đắt) → hiểu tại sao production dùng kiến trúc 2 tầng retrieve-then-rerank.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: RAG hỏi–đáp nội bộ trên tài liệu doanh nghiệp (tiếp tục phát triển từ lab)

#### 1. Hiện trạng
- **Pipeline hiện tại:** basic paragraph chunking (500 ký tự) + dense-only search (1 bi-encoder), không rerank, không enrichment, chưa có evaluation tự động.
- **Vấn đề / Bottlenecks đang gặp:** retrieval precision thấp khi câu hỏi dùng từ khác tài liệu (synonym tiếng Việt); không phát hiện được hallucination; tài liệu versioned (v2023/v2024) khiến model trả lời nhầm bản cũ; chưa có con số nào đo chất lượng nên không biết cải thiện có hiệu lực không.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical parent-child (parent 2048 / child 256) như M1 — retrieve child cho precision, trả parent cho đủ context; riêng tài liệu markdown nhiều header thì dùng thêm structure-aware để không cắt giữa bảng/list.
2. **Search retrieval:** Hybrid BM25 + Dense + RRF (k=60) như M2 — BM25 bắt đúng từ khóa chuyên ngành/mã số, Dense bắt synonym; RRF hợp nhất mà không cần chuẩn hóa điểm.
3. **Reranking:** Có — bge-reranker-v2-m3, top-20 → top-3, kèm `benchmark_reranker()` để theo dõi latency; nếu CPU quá chậm thì fallback FlashRank (<5ms).
4. **Evaluation:** RAGAS 4 metrics + `failure_analysis()` bottom-N tự động sau mỗi lần chỉnh pipeline, lưu JSON vào reports/ để so sánh giữa các phiên bản (regression test cho RAG).
5. **Enrichment:** Combined single-call mode (1 call/chunk cho summary + HyQA + contextual prepend + metadata) — tối ưu chi phí; ưu tiên contextual prepend vì Anthropic benchmark cho giảm 49% retrieval failure.

#### 3. Timeline triển khai
- **Tuần 1:** Áp dụng M1 (hierarchical chunking) + M2 (hybrid search RRF), thiết lập test set 20 câu theo 6 loại như lab (lookup, version, negation, multi-hop, numeric, ambiguous).
- **Tuần 2:** Thêm M3 rerank + M4 RAGAS vào CI chạy định kỳ; dựng dashboard so sánh naive vs production theo từng commit.
- **Tuần 3:** M5 enrichment combined mode, A/B test enrichment on/off trên test set; chốt ngưỡng chất lượng (mục tiêu ≥ 0.75 cả 4 metrics).
