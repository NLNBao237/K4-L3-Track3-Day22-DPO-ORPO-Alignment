# Thẻ mô hình — Gia sư Python gợi mở (v0, thử nghiệm)

> Bản thử nghiệm phục vụ học tập. Không dùng thay giảng viên, giáo trình hay trợ giảng.

## Tóm tắt

| | |
|---|---|
| Loại | Adapter LoRA (r=16, α=32) trên mô hình SFT của Lab 22 |
| Mô hình gốc | `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` (Qwen3-1.7B, Apache-2.0), tắt chế độ suy nghĩ |
| Bước 1 — SFT | 1.000 mẫu `saillab/alpaca-vietnamese-cleaned` (CC BY-NC), 1 epoch → `models/sft-merged` |
| Bước 2 — sở thích | 170 cặp tiếng Việt tự soạn cho lĩnh vực gia sư Python (`bonus/data/`), 30 cặp held-out |
| Loss | RPO: `loss_type=["sigmoid","sft"]`, β = 0,1 (DPO + NLL trên câu `chosen`) |
| Siêu tham số | lr 5e-5 (cosine, warmup 10%), 5 epoch, batch 1 × tích luỹ 4, `max_length` 768, seed 42 |
| Tham chiếu của DPO | chính mô hình SFT (`precompute_ref_log_probs=True`) |
| Phần cứng | RTX 3060 Laptop 6 GB, khoảng 15 phút |
| Ngôn ngữ | Tiếng Việt |
| Giấy phép sử dụng | Chỉ học tập và nghiên cứu, phi thương mại (do dữ liệu SFT là CC BY-NC) |

## Mô hình này dùng để làm gì

- Trả lời câu hỏi bài tập Python nhập môn của sinh viên năm nhất theo kiểu **gợi mở**: ghi nhận phần đã làm đúng,
  đưa gợi ý hoặc câu hỏi ngược lại, mời người học tự làm bước tiếp.
- Dùng thuật ngữ tiếng Việt theo giáo trình thay vì pha tiếng Anh.
- Khi người học đã làm xong và chỉ hỏi "đúng chưa ạ?", xác nhận trực tiếp.
- Làm ví dụ minh hoạ cho việc dùng DPO/RPO để chuyển **cách trả lời** của một mô hình nhỏ.

## Mô hình này KHÔNG dùng để làm gì

- **Không** dùng làm nguồn kiến thức đúng sai về Python. Nó có thể khẳng định sai bằng giọng tự tin.
- **Không** dùng để chấm bài, cho điểm, hay xác nhận bài làm trong thi cử, kiểm tra.
- **Không** đưa cho người học dùng mà không có người hiểu Python giám sát.
- **Không** dùng ngoài phạm vi Python nhập môn (không huấn luyện cho ngôn ngữ khác, thuật toán nâng cao, hay câu
  hỏi không phải lập trình).
- **Không** dùng cho mục đích thương mại.
- Không có bất kỳ huấn luyện nào về an toàn nội dung; trên 8 câu hỏi cố định của lab, mô hình SFT nền đã không
  từ chối dứt khoát yêu cầu nguy hiểm (xem `submission/REFLECTION.md` §4).

## Kết quả đánh giá

Trên 30 cặp held-out (không trùng câu hỏi với tập huấn luyện), từ `adapters/dpo-bonus/dpo_metrics.json`:

| Chỉ số | Giá trị |
|---|---:|
| Độ chính xác reward (chosen được ưu tiên hơn rejected) | 1,00 (30/30) |
| Reward chosen / rejected | +12,83 / −3,83 |
| Margin | +16,66 |
| Chẩn đoán | INTENDED (chosen tăng, rejected giảm) |

Hành vi khi sinh câu trả lời, trên 25 câu hỏi held-out loại `hint`, giải mã tham lam, đo bằng heuristic của
`bonus/evaluate.py` (`bonus/eval_results.json`):

| | SFT (trước) | Gia sư v2 (sau) | Câu `chosen` mẫu |
|---|---:|---:|---:|
| Đưa thẳng khối code ≥ 60 ký tự | 48% | 4% | 12% |
| Có hỏi ngược lại học sinh | 4% | 56% | 96% |
| Từ tiếng Anh thừa mỗi câu trả lời | 0,04 | 0,00 | 0,04 |
| Độ dài trung bình | 416 ký tự | 414 ký tự | 565 ký tự |

Độ dài gần như không đổi, đúng như mong muốn khi dữ liệu được cân bằng độ dài (chosen dài hơn ở 50,5% số cặp).

**Chưa đo:** độ đúng của nội dung gợi ý, và việc người học có thật sự giải được bài sau khi nhận gợi ý hay không.
Đây là hai chỉ số quan trọng nhất và chưa có.

## Hạn chế đã biết

1. **Sai kiến thức.** Trong 5 mẫu ở `demo/5-samples.md`, mẫu 4 nói code của học sinh "đã viết đúng" trong khi
   điều kiện `c == "a" or "e" or …` luôn đúng. Mô hình học được giọng gia sư nhanh hơn nhiều so với năng lực gỡ lỗi.
2. **Lộ lời giải.** Mẫu 3 đưa luôn biểu thức cốt lõi `print(i * '*')` trong phần "gợi ý".
3. **Một giọng văn.** Toàn bộ dữ liệu do một nguồn soạn; mô hình hay mở đầu bằng "Em đã…".
4. **Một lượt.** Không có dữ liệu hội thoại nhiều lượt, không biết "đã gợi ý hai lần rồi, giờ nên giải thích hẳn".
5. **Cỡ mẫu nhỏ.** 30 cặp held-out, trong đó chỉ 5 cặp `check`; các con số phần trăm ở trên có sai số lớn.
6. **Thước đo heuristic** (đếm dấu hỏi, đếm khối code, danh sách từ tiếng Anh) có thể bị mô hình thoả mãn một cách
   hình thức.
7. **Thiên lệch dữ liệu.** Câu hỏi là tưởng tượng về sinh viên Việt Nam xưng "em", viết bằng tiếng Việt chuẩn; chưa
   có tiếng lóng, viết tắt, không dấu, hay câu hỏi lẫn tiếng Anh như tin nhắn thật.
8. **Quyền riêng tư:** dữ liệu không chứa thông tin cá nhân thật; demo chạy cục bộ, không gửi câu hỏi đi đâu.

## Dữ liệu huấn luyện

200 cặp `(prompt, chosen, rejected)` trên 20 chủ đề Python nhập môn; 170 cặp `hint` (chosen = gợi mở, rejected =
đưa thẳng code đúng kèm lời giải thích pha tiếng Anh) và 30 cặp `check` (chosen = xác nhận trực tiếp, rejected =
từ chối xác nhận hoặc giảng giải chung chung). Mọi cặp có tỉ lệ độ dài rejected/chosen trong khoảng 0,7 – 1,4.

**Nguồn gốc:** các cặp do Claude Code (Anthropic) soạn theo đặc tả sư phạm của người thực hiện, không dịch từ
UltraFeedback. Code trong dữ liệu được biên dịch và chạy thử tự động trên Python 3.11. Chưa có người rà soát thủ
công toàn bộ.

## Cách dùng

```bash
python bonus/demo/serve.py              # cần models/sft-merged (NB1) + adapters/dpo-bonus (bonus/train.py)
```

Hoặc trong Python:

```python
import unsloth
from lab22 import modeling as MD
model, tokenizer = MD.load_model("adapters/dpo-bonus")
print(MD.generate(model, tokenizer, ["Giải giúp em bài tính tổng các số chẵn từ 1 đến n ạ."])[0])
```

## Nhật ký quy trình vibe coding

Phần lớn mã và dữ liệu của thư mục này do Claude Code viết; người thực hiện chọn lĩnh vực và hướng chạy.

- **Câu lệnh hiệu quả nhất:** đặc tả soạn dữ liệu có ràng buộc đo được thay vì mô tả chung chung: "rejected phải
  dài từ 0,7 đến 1,4 lần chosen", "code trong rejected phải chạy đúng", "khoảng 15% cặp là tình huống học sinh đã
  làm xong", kèm yêu cầu tự viết script kiểm tra rồi mới báo xong. Bản nháp đầu có chosen dài hơn ở 76 – 85% số
  cặp; nhờ ràng buộc có số nên lỗi này bị bắt và sửa về 50,5% trước khi huấn luyện.
- **Câu lệnh thất bại:** "huấn luyện DPO trên dữ liệu này giống NB3". Kết quả đẹp trên giấy (độ chính xác reward
  1,00, margin +7,18) nhưng câu trả lời sinh ra không đổi văn phong: tỉ lệ hỏi ngược lại vẫn 4%. Chỉ khi đọc trực
  tiếp 5 mẫu sinh ra mới thấy, rồi mới chuyển sang RPO. Bài học: với mô hình căn chỉnh, phải đọc đầu ra thật, số
  reward không đủ.
