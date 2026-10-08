# 5 mẫu trước/sau DPO (câu hỏi held-out, giải mã tham lam)

## Mẫu 1 — `b030` · tep-tin · hint

**Học sinh hỏi:**

Em muốn lấy dòng đầu tiên của tệp mà nó chỉ in ra đúng 1 chữ cái ạ
```python
with open("truyen.txt", encoding="utf-8") as f:
    noi_dung = f.read()
print(noi_dung[0])
```

**Trước (SFT):**

Đây là một ví dụ về cách lấy dòng đầu tiên của một tệp trong Python: ```python with open("truyen.txt", encoding="utf-8") as f: noi_dung = f.read() print(noi_dung[0]) ``` Trong ví dụ này, chúng ta mở tệp "truyen.txt" và đọc toàn bộ nội dung của nó vào biến `nội dung`. Sau đó, chúng ta chỉ ra một chữ cái đầu tiên của `nội dung` bằng cách sử dụng `nội dung[0]`. Điều này sẽ trả về một chuỗi chứa một chữ cái đầu tiên của tệp.

**Sau (SFT + DPO gia sư):**

Em đã đọc đúng rồi, chỉ cần cắt ra một ký tự là được. Em thử thêm `print(noi_dung[0].isalpha())` để kiểm tra xem ký tự đó có phải chữ cái hay không. Em cũng có thể dùng `f.readline()` thay vì `f.read()` để chỉ đọc một dòng mà không cần biết độ dài. Em viết thử cả hai cách và so sánh xem có khác nhau ở đâu nhé.

---

## Mẫu 2 — `a087` · tu-dien · hint

**Học sinh hỏi:**

Đảo ngược từ điển: {"a": 1, "b": 2} thành {1: "a", 2: "b"}. Giải giúp em với ạ.

**Trước (SFT):**

Đây là cách đảo ngược từ điển: {"a": 1, "b": 2} thành {1: "a", 2: "b"}.

**Sau (SFT + DPO gia sư):**

Em đã có một từ điển và một từ điển đích rồi. Em nhìn xem hai từ điển đó có gì giống nhau: mỗi từ điển đều có hai phần tử, mỗi phần tử có một khóa và một giá trị. Em so sánh hai phần tử của hai từ điển: khóa của từ điển đích là gì, giá trị của nó là gì? Em viết một vòng lặp `for` duyệt qua từng khóa và giá trị của từ điển ban đầu, rồi đặt chúng vào từ điển đích theo đúng thứ tự em đã suy ra từ hai phần tử đầu tiên. Em thử với hai từ điển của mình và sửa lại dòng `d2[1] = "a"` nếu cần nhé.

---

## Mẫu 3 — `a048` · vong-lap-for · hint

**Học sinh hỏi:**

Đề in tam giác sao n dòng:
*
**
***
Em dùng for mà chỉ in được mỗi dòng một sao.

**Trước (SFT):**

Đây là một cách để in tam giác sao với vòng lặp for trong Python: ```python for i in range(1, 4): print('*' * i) ```

**Sau (SFT + DPO gia sư):**

Em đã nghĩ đến vòng lặp rồi, chỉ còn là cách in mỗi dòng. Em thử viết vòng lặp với `range(1, 4)` và in `print(i * '*')` xem có ra 3 dòng sao mong đợi không. Em quan sát xem số sao của mỗi dòng có mối liên hệ với số dòng, rồi viết hàm chung với tham số `n` nhé.

---

## Mẫu 4 — `b097` · bai-tap-tong-hop · hint

**Học sinh hỏi:**

Em đếm số nguyên âm trong một chuỗi mà nó đếm luôn cả phụ âm, chuỗi "python" ra 6 ạ
```python
s = "python"
dem = 0
for c in s:
    if c == "a" or "e" or "i" or "o" or "u":
        dem += 1
print(dem)
```

**Trước (SFT):**

Chuỗi "python" có 2 nguyên âm: "o" và "y".

**Sau (SFT + DPO gia sư):**

Em đã viết đúng, và kết quả "python" cho 6 là đúng. Em nhìn lại dòng `if c == "a" or "e" or "i" or "o" or "u"`: mỗi lần gặp chữ cái nào thì nó đều được xét, và chỉ một trong số đó mới được cộng. Em thử viết lại với `or` thay vì `or "e"`, rồi chạy xem có khác không. Em cũng có thể viết gọn bằng `if c in {'a', 'e', 'i', 'o', 'u'}`. Em thử một chuỗi có chữ cái khác, ví dụ "abc", và đếm xem nó cho mấy.

---

## Mẫu 5 — `b099` · bai-tap-tong-hop · check

**Học sinh hỏi:**

Em viết hàm đếm số từ trong một câu, thử với "em học lập trình Python" ra 5. Em làm vậy ổn không ạ?
```python
def dem_tu(cau):
    return len(cau.split())
```

**Trước (SFT):**

Đây là một hàm đếm số từ trong một câu, thử với "em học lập trình Python" ra 5. Em làm vậy ổn không ạ?

**Sau (SFT + DPO gia sư):**

Đúng rồi em, hàm này đếm đúng 5 từ trong "em học lập trình Python". Cách làm là tách chuỗi thành danh sách bằng `split()` và lấy độ dài của danh sách đó. Em có thể kiểm tra thêm bằng `print(len("em học lập trình".split()))` để thấy kết quả. Nếu muốn chính xác hơn, em có thể in ra danh sách các từ để xem có 5 hay 6. Hàm này không có lỗi gì, và em có thể mở rộng thêm nếu muốn đếm số chữ cái hay chữ hoa.

---
