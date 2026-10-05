# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** HongPhi  
**Khóa:** K4 - Track 3B  

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8083 | **0.9167** | +0.1083 |
| Answer Relevancy | 0.6954 | **0.7565** | +0.0611 |
| Context Precision | 0.9250 | **0.9917** | +0.0667 |
| Context Recall | 0.9000 | **0.9333** | +0.0333 |

**Kết luận:** Production (hierarchical chunking + enrichment + hybrid search + rerank) thắng baseline trên **cả 4 metrics**. Đóng góp lớn nhất đến từ: (1) retrieve child → **return parent** cho LLM context đầy đủ, (2) rerank top-20→top-3 loại bỏ nhiễu, (3) `temperature=0.1` + prompt chặt hạn chế bịa.

## Bottom-5 Failures

### #1
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 ngày + 3 ngày thâm niên (9÷3) = 18 ngày; lương Senior (P3–P4) 20–35 triệu VNĐ/tháng.
- **Got:** Trả lời thiếu một trong hai mảng (phép năm hoặc lương).
- **Worst metric:** answer_relevancy = 0.0 (avg 0.625)
- **Error Tree:** Output sai → Context đúng? → **Context chỉ chứa parent của 1 document** (phép năm hoặc lương) → Query OK? (hybrid search có cả 2 chunk trong top-20) → Rerank top-3 nhưng mỗi child chỉ kéo về parent của 1 chủ đề.
- **Root cause:** Câu hỏi multi-hop cần thông tin từ 2 documents; kiến trúc "1 child → 1 parent" chỉ mở rộng context theo 1 nhánh nên LLM thiếu evidence cho nửa còn lại.
- **Suggested fix:** Tăng RERANK_TOP_K lên 5–6 cho query multi-hop; hoặc query decomposition (tách "số ngày phép" + "lương Senior") rồi gộp context trước khi sinh câu trả lời.

### #2
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày, phí 2%/tháng pro-rata ≈ 50.000 VNĐ.
- **Got:** LLM đưa ra con số tính sai lệch (ví dụ 300.000đ).
- **Worst metric:** faithfulness = 0.33 (avg 0.686)
- **Error Tree:** Output sai → Context đúng? (context có quy định 2%/tháng nhưng **không có công thức pro-rata theo ngày**) → LLM tự suy diễn phép tính ngoài evidence → hallucination về con số.
- **Root cause:** Policy chỉ nêu "%/tháng"; câu hỏi đòi tính toán đại số. LLM "tính giúp" số không có trong context → faithfulness sụp dù retrieval đúng.
- **Suggested fix:** Prompt yêu cầu trích nguyên văn mệnh đề trước khi tính; hoặc cho LLM trả lời dạng "công thức + số liệu trích từ context" thay vì kết quả cuối; thêm step calculator có kiểm soát nếu cần.

### #3
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** KHÔNG — chỉ có BHXH bắt buộc.
- **Got:** Đúng nội dung ("KHÔNG...").
- **Worst metric:** answer_relevancy = 0.0 (avg 0.75)
- **Error Tree:** Output sai? → Output đúng → Judge sinh câu hỏi giả định từ answer để so similarity với câu hỏi gốc → câu trả lời phủ định ("KHÔNG, ...") sinh ra câu hỏi lệch sắc câu gốc → similarity thấp.
- **Root cause:** False negative của RAGAS answer_relevancy với câu trả lời phủ định (negation) — metric sinh câu hỏi từ answer rồi so embedding, không phù hợp với dạng Q&A yes/no.
- **Suggested fix:** Ghi nhận là hạn chế của metric (không phải lỗi pipeline); nâng cấp RAGAS ≥0.2 hoặc tự định nghĩa metric polarity-match cho câu hỏi yes/no.

### #4
- **Question:** Có cần kích hoạt xác thực đa yếu tố (MFA) không?
- **Expected:** Có — bắt buộc cho email, VPN và hệ thống nội bộ (chính sách v2.0).
- **Got:** Trả lời "Có" nhưng thiếu chi tiết phạm vi.
- **Worst metric:** context_recall = 0.5 (avg 0.810)
- **Error Tree:** Context sai? → Context có chunk quy định MFA nhưng **thiếu câu liệt kê phạm vi** (email/VPN/hệ thống nội bộ) nằm ở vị trí khác trong tài liệu v2 → Chunk đúng có trong top-k? (có) → Parent 2048 ký tự không bao phủ hết các mệnh đề liên quan.
- **Root cause:** Ground truth gộp thông tin từ 2 vị trí trong cùng 1 tài liệu; hierarchical parent không trọn được cả hai.
- **Suggested fix:** Tăng HIERARCHICAL_PARENT_SIZE hoặc dùng parent theo section (structure-aware) để các mệnh đề về MFA nằm chung một parent.

### #5
- **Question:** Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành — phải hoàn trả bao nhiêu?
- **Expected:** 100% = 25.000.000 VNĐ (cam kết tối thiểu 1 năm sau khóa học).
- **Got:** Đúng số 25 triệu nhưng kèm suy diễn chưa được context bảo chứng đầy đủ.
- **Worst metric:** faithfulness = 0.5 (avg 0.830)
- **Error Tree:** Output sai? → Claim đúng (hoàn 100%) + claim suy diễn ("vì nghỉ trước cam kết 1 năm") mà context chỉ nêu quy định rời rạc → Judge đánh dấu claim suy diễn là unsupported.
- **Root cause:** LLM kết luận logic đúng từ evidence nhưng cách diễn đạt không khớp nguyên văn; RAGAS faithfulness phạt cả claim suy diễn.
- **Suggested fix:** Prompt: "nếu phải suy diễn, ghi rõ 'suy ra từ quy định: <trích dẫn>'"; cân nhắc ngưỡng tolerance khi đọc faithfulness cho câu hỏi tính-toán/suy-diễn.

## Case Study (cho presentation)

**Question chọn phân tích:** #2 (tạm ứng 15 triệu, phạt bao nhiêu?)

**Error Tree walkthrough:**
1. Output đúng? → KHÔNG (con số phạt sai)
2. Context đúng? → CÓ về quy định (2%/tháng) nhưng THIẾU công thức pro-rata → retrieval không phải thủ phạm
3. Query rewrite OK? → CÓ (BM25 + dense đều tìm đúng tài liệu tam_ung.md)
4. Fix ở bước: **Generation** — siết prompt (bắt trích dẫn mệnh đề, khai báo giả định khi tính), không phải retrieval.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Prompt generation: thêm "Liệt kê các mệnh đề context sử dụng trước khi trả lời" để ép LLM grounding claim-by-claim (nhắm thẳng vào 2/5 failure là faithfulness).
- RERANK_TOP_K 3 → 5 cho các query multi-hop (nhắm vào #1, #4 — thiếu evidence từ document thứ hai).

---

## Latency Breakdown (bonus)

Máy test: CPU-only (GPU GTX 1650 Ti 4GB không dùng được cho 2 model lớn), Python 3.11, bge-m3 + bge-reranker-v2-m3 chạy CPU.

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| Load documents (26 files) | ~0.1s | 25 .md + 1 PDF text layer (2 PDF scan bị bỏ qua) |
| M1 Hierarchical chunking (104 children) | <0.1s | parent 2048 / child 256 ký tự |
| M5 Enrichment — 104 chunks × 1 LLM call | **331.1s** (~3.2s/chunk) | Chi phí lớn nhất; chỉ xảy ra lúc build index |
| M2 Indexing BM25 + Dense (bge-m3 CPU) | 31.0s | ~0.3s/chunk |
| M3 Reranker load (lần đầu) | ~15s | Sau đó cached dùng chung trong process |
| M3 Rerank 1 query (top-20 → top-3) | **~4,547ms** | CPU, batch 20 pairs |
| M2 Hybrid search 1 query (BM25 + Dense + RRF) | ~150ms | |
| LLM answer generation 1 query | ~2–4s | openai/gpt-4o-mini qua gateway |
| 20 queries end-to-end (search + rerank + answer) | ~250s (~12.5s/query) | |
| M4 RAGAS eval (20 câu × 4 metrics) | 37.7s | + ~20s load bge-m3 làm embeddings |
| **Tổng pipeline** | **660.7s** | |

**Nhận xét:** Rerank chiếm ~36% thời gian mỗi query trên CPU — đánh đổi đáng giá vì context_precision tăng 0.925 → 0.99. Nếu cần latency thấp hơn cho production: (1) rerank chỉ top-10 candidates, (2) FlashRank (<5ms) thay bge-reranker với chất lượng giảm nhẹ, (3) chạy trên GPU.
