# Gia sư Python gợi mở (v0, thử nghiệm)

**Người thực hiện:** Nguyễn Lê Ngọc Bảo (A20-K4), làm cùng Claude Code. Phần nào do AI soạn được ghi rõ ở mục
"Dữ liệu" và trong [`MODEL-CARD.md`](MODEL-CARD.md).

Dựa trên gợi ý số 1 của `BONUS-CHALLENGE.md`: dùng dữ liệu sở thích để dạy mô hình **gợi mở từng bước** thay vì
đưa thẳng đáp án.

## 1. Người dùng

Sinh viên năm nhất (hoặc học sinh THPT) Việt Nam đang học nhập môn lập trình bằng Python, tự làm bài tập ở nhà
và hay bí ở các bài kinh điển: tính tổng số chẵn, đếm nguyên âm, in tam giác sao, đọc tệp, lỗi `IndexError`,
`TypeError`… Các bạn này thường dán nguyên đề bài hoặc đoạn code lỗi kèm câu "giải giúp em với ạ".

## 2. Kiến thức chuyên môn đem vào

Điều cốt lõi là phân biệt **câu trả lời có tính sư phạm** với **câu trả lời đưa thẳng lời giải**. Một chatbot
thông thường sẽ trả về ngay khối code hoàn chỉnh. Người học chép, chạy được, và không học được gì. Bộ dữ liệu này
mã hoá ba quy tắc:

1. **Gợi ý, không giải hộ.** Ghi nhận phần em đã làm đúng, đưa một hoặc hai gợi ý hay một câu hỏi ngược lại, có
   thể kèm một ví dụ nhỏ tương tự, rồi mời em tự làm bước tiếp theo. Không đưa lời giải hoàn chỉnh của đúng bài
   được hỏi.
2. **Dùng thuật ngữ tiếng Việt theo giáo trình**: biến, vòng lặp, danh sách, từ điển, chỉ số, tham số, đệ quy,
   điều kiện dừng… thay cho kiểu nói pha tiếng Anh ("dùng loop để iterate qua list rồi check condition").
3. **Bối cảnh quyết định khi nào trả lời thẳng.** Nếu em đã tự làm xong và chỉ hỏi "đúng chưa ạ?", gia sư phải
   xác nhận trực tiếp (hoặc chỉ đúng một lỗi cụ thể), không vòng vo hỏi ngược. 30/200 cặp thuộc loại này
   (`kind = "check"`), trong đó câu bị loại chính là kiểu "từ chối xác nhận, bắt em tự tìm".

## 3. Mục tiêu ứng dụng

Một mô hình nhỏ chạy được trên máy cá nhân, nhận câu hỏi bài tập Python bằng tiếng Việt và trả lời như một trợ
giảng: chỉ hướng, hỏi lại, để người học tự viết code. Không nhằm thay giáo trình hay giảng viên.

## 4. Sản phẩm đầu ra

| Thành phần | Đường dẫn |
|---|---|
| 200 câu hỏi theo 20 chủ đề | [`data/prompts.jsonl`](data/prompts.jsonl) |
| 200 cặp sở thích (chosen / rejected) | [`data/pairs.parquet`](data/pairs.parquet), bản gốc dễ đọc ở [`data/raw/`](data/raw) |
| Tập huấn luyện 170 cặp / held-out 30 cặp (không trùng câu hỏi) | `data/train.parquet`, `data/eval.parquet` |
| Script dựng dữ liệu, huấn luyện, đánh giá | [`build_data.py`](build_data.py), [`train.py`](train.py), [`evaluate.py`](evaluate.py) |
| Adapter LoRA (trọng số nằm ngoài git) | `adapters/dpo-bonus/` (số liệu: `dpo_metrics.json`) |
| Bản demo dòng lệnh | [`demo/serve.py`](demo/serve.py) |
| 5 mẫu trước/sau đầy đủ | [`demo/5-samples.md`](demo/5-samples.md) |
| Thẻ mô hình | [`MODEL-CARD.md`](MODEL-CARD.md) |

### Chạy thử

Cần đã chạy NB1 của lab (để có `models/sft-merged/`) và có GPU CUDA ≥ 6 GB.

```bash
python bonus/build_data.py      # dựng lại parquet từ data/raw (vài giây, không cần GPU)
python bonus/train.py           # ~15 phút trên RTX 3060 6 GB → adapters/dpo-bonus/
python bonus/evaluate.py        # so sánh trước/sau trên 30 câu held-out
python bonus/demo/serve.py      # trò chuyện với gia sư trong cửa sổ dòng lệnh
```

`python bonus/demo/serve.py --baseline` chạy mô hình SFT chưa căn chỉnh để so sánh.

## Dữ liệu

- **Phân bố:** 20 chủ đề × 10 cặp (biến và kiểu dữ liệu, chuỗi, nhập xuất, rẽ nhánh, `for`, `while`, danh sách,
  tuple và tập hợp, từ điển, hàm, gỡ lỗi, đệ quy, tệp, danh sách rút gọn, sắp xếp và tìm kiếm, lớp và đối tượng,
  mô-đun, ngoại lệ, độ phức tạp, bài tập tổng hợp). 170 cặp `hint`, 30 cặp `check`.
- **Cân bằng độ dài có chủ đích.** Bài học từ NB2 của lab: nếu `chosen` thường dài hơn, DPO học "viết dài". Ở đây
  `chosen` dài hơn `rejected` trong đúng 50,5% số cặp (trung vị 558 so với 535 ký tự), và tỉ lệ độ dài của mọi cặp
  nằm trong khoảng 0,7 – 1,4. Mô hình buộc phải học văn phong chứ không thể dựa vào độ dài.
- **Câu bị loại vẫn đúng về mặt lập trình.** Code trong `rejected` là code chạy đúng; thứ bị chê là cách dạy và
  cách dùng từ. Như vậy mô hình không học nhầm "code đúng là xấu".
- **Nguồn gốc (nói thật):** cả 200 cặp do **Claude Code soạn** theo đặc tả ở trên, không phải dịch từ
  UltraFeedback và không lấy từ tin nhắn thật của sinh viên. Các đoạn code trong `rejected` và trong câu hỏi đã
  được biên dịch và chạy thử tự động trên Python 3.11. **Chưa có** giảng viên hay sinh viên thật nào rà soát toàn
  bộ 200 cặp. Đây là điểm yếu lớn nhất so với tiêu chí "ít nhất 100 cặp do chính bạn tạo" của đề bài.

## Kết quả

Số liệu từ `adapters/dpo-bonus/dpo_metrics.json`, `metrics-v1-dpo.json` và `eval_results*.json`. Các chỉ số hành
vi được đo bằng heuristic trong `evaluate.py` trên 25 câu hỏi held-out loại `hint`, giải mã tham lam.

| | Đưa thẳng khối code (≥ 60 ký tự) | Có hỏi ngược lại | Từ tiếng Anh thừa / câu | Độ chính xác reward held-out | Margin held-out |
|---|---:|---:|---:|---:|---:|
| SFT (trước) | 48% | 4% | 0,04 | — | — |
| v1: DPO, lr 2e-5, 3 epoch | 28% | 4% | 0,24 | 1,00 (30/30) | +7,18 |
| **v2: RPO, lr 5e-5, 5 epoch** | **4%** | **56%** | **0,00** | 1,00 (30/30) | +16,66 |
| *Mốc: câu `chosen` do người soạn viết* | *12%* | *96%* | *0,04* | | |
| *Mốc: câu `rejected`* | *100%* | *0%* | *8,12* | | |

Điều đáng học nhất nằm ở dòng v1. Chỉ dùng DPO, mô hình **xếp hạng đúng cả 30/30 cặp held-out** nhưng câu trả lời
nó tự sinh ra hầu như không đổi: vẫn mở đầu bằng "Đây là…" kiểu Alpaca và không hề hỏi lại học sinh. DPO chỉ tối
ưu *hiệu* log-xác suất giữa hai câu cho sẵn; ở v1 phần lớn margin đến từ việc đẩy câu `rejected` xuống (reward
−4,26) chứ không phải kéo câu `chosen` lên đủ để nó trở thành câu mà mô hình tự viết ra. Bản v2 thêm số hạng NLL
trên câu `chosen` (RPO, `loss_type=["sigmoid","sft"]`): reward của `chosen` trên held-out tăng từ +2,92 lên +12,83
và văn phong gợi mở mới thật sự xuất hiện khi sinh.

Với các câu `check` (chỉ 5 câu held-out, quá ít để kết luận): v2 xác nhận trực tiếp ở mẫu 5 bên dưới, và tỉ lệ
hỏi ngược lại giảm từ 2/5 xuống 1/5.

## 5 lượt tương tác mẫu (v2, câu hỏi held-out)

Câu trả lời của bản SFT cho cùng 5 câu hỏi nằm ở [`demo/5-samples.md`](demo/5-samples.md).

**Mẫu 1** (`b030`, hint)

> Em muốn lấy dòng đầu tiên của tệp mà nó chỉ in ra đúng 1 chữ cái ạ
> ```python
> with open("truyen.txt", encoding="utf-8") as f:
>     noi_dung = f.read()
> print(noi_dung[0])
> ```

**Gia sư:** Em đã đọc đúng rồi, chỉ cần cắt ra một ký tự là được. Em thử thêm `print(noi_dung[0].isalpha())` để kiểm tra xem ký tự đó có phải chữ cái hay không. Em cũng có thể dùng `f.readline()` thay vì `f.read()` để chỉ đọc một dòng mà không cần biết độ dài. Em viết thử cả hai cách và so sánh xem có khác nhau ở đâu nhé.

**Mẫu 2** (`a087`, hint)

> Đảo ngược từ điển: {"a": 1, "b": 2} thành {1: "a", 2: "b"}. Giải giúp em với ạ.

**Gia sư:** Em đã có một từ điển và một từ điển đích rồi. Em nhìn xem hai từ điển đó có gì giống nhau: mỗi từ điển đều có hai phần tử, mỗi phần tử có một khóa và một giá trị. Em so sánh hai phần tử của hai từ điển: khóa của từ điển đích là gì, giá trị của nó là gì? Em viết một vòng lặp `for` duyệt qua từng khóa và giá trị của từ điển ban đầu, rồi đặt chúng vào từ điển đích theo đúng thứ tự em đã suy ra từ hai phần tử đầu tiên. Em thử với hai từ điển của mình và sửa lại dòng `d2[1] = "a"` nếu cần nhé.

**Mẫu 3** (`a048`, hint)

> Đề in tam giác sao n dòng:
> *
> **
> ***
> Em dùng for mà chỉ in được mỗi dòng một sao.

**Gia sư:** Em đã nghĩ đến vòng lặp rồi, chỉ còn là cách in mỗi dòng. Em thử viết vòng lặp với `range(1, 4)` và in `print(i * '*')` xem có ra 3 dòng sao mong đợi không. Em quan sát xem số sao của mỗi dòng có mối liên hệ với số dòng, rồi viết hàm chung với tham số `n` nhé.

**Mẫu 4** (`b097`, hint)

> Em đếm số nguyên âm trong một chuỗi mà nó đếm luôn cả phụ âm, chuỗi "python" ra 6 ạ
> ```python
> s = "python"
> dem = 0
> for c in s:
>     if c == "a" or "e" or "i" or "o" or "u":
>         dem += 1
> print(dem)
> ```

**Gia sư:** Em đã viết đúng, và kết quả "python" cho 6 là đúng. Em nhìn lại dòng `if c == "a" or "e" or "i" or "o" or "u"`: mỗi lần gặp chữ cái nào thì nó đều được xét, và chỉ một trong số đó mới được cộng. Em thử viết lại với `or` thay vì `or "e"`, rồi chạy xem có khác không. Em cũng có thể viết gọn bằng `if c in {'a', 'e', 'i', 'o', 'u'}`. Em thử một chuỗi có chữ cái khác, ví dụ "abc", và đếm xem nó cho mấy.

**Mẫu 5** (`b099`, check)

> Em viết hàm đếm số từ trong một câu, thử với "em học lập trình Python" ra 5. Em làm vậy ổn không ạ?
> ```python
> def dem_tu(cau):
>     return len(cau.split())
> ```

**Gia sư:** Đúng rồi em, hàm này đếm đúng 5 từ trong "em học lập trình Python". Cách làm là tách chuỗi thành danh sách bằng `split()` và lấy độ dài của danh sách đó. Em có thể kiểm tra thêm bằng `print(len("em học lập trình".split()))` để thấy kết quả. Nếu muốn chính xác hơn, em có thể in ra danh sách các từ để xem có 5 hay 6. Hàm này không có lỗi gì, và em có thể mở rộng thêm nếu muốn đếm số chữ cái hay chữ hoa.


## Bản thử nghiệm này chưa làm được gì

- **Nội dung gợi ý có thể sai.** Mô hình nền chỉ 1,7 tỉ tham số. DPO/RPO đổi được *cách nói* nhưng không thêm
  kiến thức: ở mẫu 4 mô hình nói "em đã viết đúng" trong khi code có lỗi thật (`c == "a" or "e"` luôn đúng), và
  gợi ý sau đó khó hiểu. Một gia sư nói sai bằng giọng tự tin còn tệ hơn một chatbot đưa code đúng. Chưa nên đưa
  cho người học thật dùng mà không có người kiểm tra.
- **Vẫn có lúc đưa gần hết lời giải.** Ở mẫu 3, gợi ý chứa luôn `print(i * '*')`, tức là phần cốt lõi của bài.
- **Thước đo là heuristic.** "Có dấu hỏi" không có nghĩa là câu hỏi hay; "không có khối code" không có nghĩa là
  gợi ý đúng. Chưa có giảng viên chấm, chưa thử với sinh viên thật. Cách đánh giá đúng là cho 5–10 sinh viên năm
  nhất dùng thử và xem các bạn có tự giải được bài sau khi nhận gợi ý hay không.
- **Dữ liệu do AI soạn**, 200 cặp, một giọng văn duy nhất, câu hỏi là tưởng tượng chứ không phải tin nhắn thật.
  Mô hình có thể chỉ học bắt chước giọng văn đó (ví dụ luôn mở đầu "Em đã…").
- **Chỉ một lượt hỏi–đáp.** Gia sư thật cần hội thoại nhiều lượt: nhớ gợi ý đã đưa, và đưa đáp án khi học sinh đã
  thử vài lần mà vẫn bí. Dữ liệu chưa có tình huống này.
- **Chưa có GGUF riêng** cho adapter này; demo cần GPU. `serve.py --gguf` đã sẵn cho khi có file.
- **Giấy phép:** mô hình gốc Qwen3-1.7B (Apache-2.0); dữ liệu SFT `saillab/alpaca-vietnamese-cleaned` là CC BY-NC,
  nên adapter này chỉ dùng cho học tập và nghiên cứu, không dùng thương mại.
