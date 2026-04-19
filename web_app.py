__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
import os
import time
from google import genai
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------- 1. LOAD SYSTEM -----------------
@st.cache_resource
def load_ai_system():
    # ✅ LẤY API KEY TỪ ENV (AN TOÀN)
    api_key = st.secrets["GOOGLE_API_KEY"]
    if not api_key:
        st.error("❌ Chưa thiết lập GOOGLE_API_KEY")
        st.stop()
    

    client = genai.Client(api_key=api_key)

    # Vector DB
    embed_model = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )
    vector_db = Chroma(
        persist_directory="./my_vector_db",
        embedding_function=embed_model
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 7})

    return client, retriever


client, retriever = load_ai_system()

# ----------------- 2. HÀM GỌI AI (CHỐNG 503) -----------------
def call_gemini(prompt, model="gemini-2.5-flash"):
    for i in range(5):
        try:
            res = client.models.generate_content(
                model=model,
                contents=prompt
            )
            return res.text
        except Exception as e:
            print(f"Lỗi lần {i+1}: {e}")
            time.sleep(2)
    return "⚠️ AI đang quá tải, vui lòng thử lại sau!"

# ----------------- 3. DATA QUIZ -----------------
quiz_data = [
    {
        "question": "Câu 1: Người điều khiển xe mô tô hai bánh có được phép sử dụng ô (dù) khi đang lái xe không?",
        "options": ["A. Có, nhưng chỉ khi trời mưa nhỏ.", "B. Không, tuyệt đối cấm.", "C. Có, chỉ dành cho người ngồi sau.", "D. Tùy thuộc vào quy định của từng địa phương."],
        "answer": "B",
        "explanation": "Theo Khoản 3 Điều 30 Luật Giao thông đường bộ 2008, cấm người lái xe mô tô hai bánh sử dụng ô."
    },
    {
        "question": "Câu 2: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh1.png", 
        "options": [
            "A. Xe tải, xe khách, xe con, mô tô.", 
            "B. Xe tải, mô tô, xe khách, xe con.", 
            "C. Xe khách, xe tải, xe con, mô tô.", 
            "D. Mô tô, xe khách, xe tải, xe con."
        ],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Đường ưu tiên:** Dựa vào biển báo và biển phụ chỉ hướng đường ưu tiên (nét đậm), **Xe tải** và **Mô tô** đang nằm trên đoạn đường ưu tiên nên được quyền đi trước.\n2. **Quyền ưu tiên hướng đi:**\n- Trên đường ưu tiên: **Xe tải** đi thẳng được đi trước, **Mô tô** rẽ trái đi sau.\n- Trên đường không ưu tiên: **Xe khách** đi thẳng đi trước, **Xe con** rẽ trái phải nhường đường đi cuối cùng.\n\n✅ **Thứ tự đúng:** Xe tải -> Mô tô -> Xe khách -> Xe con."
    },
    {
        "question": "Câu 3: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh2.png", 
        "options": [
            "A. Xe công an, xe con, xe tải, xe khách.", 
            "B. Xe công an, xe khách, xe con, xe tải.", 
            "C. Xe công an, xe tải, xe khách, xe con.", 
            "D. Xe con, xe công an, xe tải, xe khách."
        ],
        "answer": "A",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Xe ưu tiên:** Xe công an đang làm nhiệm vụ nên được quyền đi đầu tiên.\n2. **Đường ưu tiên:** Xe tải (và xe con) nằm trên đường ưu tiên (biển báo tam giác hướng lên). Xe khách nằm trên đường không ưu tiên (biển tam giác ngược) nên phải nhường đường, đi cuối cùng.\n3. **Hướng không vướng:** Lúc này xét xe con và xe tải. Khi xe công an đã đi khỏi, ngã tư bên phải của xe con là đường trống (không vướng). Trong khi đó bên phải của xe tải đang vướng xe con. Vậy xe con đi trước xe tải.\n\n✅ **Thứ tự đúng:** Xe công an -> Xe con -> Xe tải -> Xe khách."
    },
    {
        "question": "Câu 4: Theo hướng mũi tên, thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh3.png", 
        "options": [
            "A. Xe tải, xe công an, xe khách, xe con.", 
            "B. Xe công an, xe khách, xe con, xe tải.", 
            "C. Xe công an, xe con, xe tải, xe khách.", 
            "D. Xe công an, xe tải, xe khách, xe con."
        ],
        "answer": "D",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Xe ưu tiên:** Xe công an làm nhiệm vụ đi đầu tiên.\n2. **Đường ưu tiên:** Xe tải nằm trên đường ưu tiên (biển báo hình thoi) nên được đi thứ hai.\n3. **Hướng đi:** Xe khách và xe con nằm trên đường không ưu tiên (biển tam giác ngược). Áp dụng quy tắc hướng đi: Đi thẳng đi trước, rẽ trái nhường đường. Vậy xe khách đi thẳng được đi trước xe con rẽ trái.\n\n✅ **Thứ tự đúng:** Xe công an -> Xe tải -> Xe khách -> Xe con."
    },
    {
        "question": "Câu 5: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh4.png", 
        "options": [
            "A. Xe tải, xe con, mô tô.", 
            "B. Xe con, xe tải, mô tô.", 
            "C. Mô tô, xe con, xe tải.", 
            "D. Xe con, mô tô, xe tải."
        ],
        "answer": "C",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Đường đồng cấp:** Không có xe ưu tiên, không có biển báo đường ưu tiên.\n2. **Quyền bên phải không vướng:** \n- Xét tại ngã tư, nhánh đường bên dưới đang trống.\n- Suy ra, phía bên tay phải của **Mô tô** không có xe $\\rightarrow$ Mô tô được đi đầu tiên.\n- Sau khi Mô tô đi, bên phải của **Xe con** trống $\\rightarrow$ Xe con đi thứ 2.\n- Cuối cùng là **Xe tải**.\n\n✅ **Thứ tự đúng:** Mô tô -> Xe con -> Xe tải."
    },
    {
        "question": "Câu 6: Xe nào phải nhường đường trong trường hợp này?",
        "image": "images/sahinh5.png", 
        "options": [
            "A. Xe con.", 
            "B. Xe tải."
        ],
        "answer": "A",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Quy tắc vòng xuyến:** Tại ngã tư giao nhau có đặt biển báo hiệu đi theo vòng xuyến, người lái xe phải nhường đường cho xe đi đến từ bên **trái**.\n2. **Áp dụng:** Xe tải đang ở trong vòng xuyến (đến từ bên trái của xe con). Xe con đang ở ngoài chuẩn bị tiến vào vòng xuyến nên bắt buộc phải nhường đường.\n\n✅ **Kết luận:** Xe con phải nhường đường."
    },
    {
        "question": "Câu 7: Trường hợp này xe nào được quyền đi trước?",
        "image": "images/sahinh6.png", 
        "options": [
            "A. Mô tô.", 
            "B. Xe con."
        ],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Biển báo hiệu:** Quan sát thấy phía trước mặt người điều khiển mô tô có biển báo hiệu lệnh **STOP** (Dừng lại).\n2. **Quy tắc:** Theo Luật Giao thông đường bộ, khi gặp biển báo STOP, tất cả các phương tiện đều phải dừng lại và nhường đường cho xe đi trên đường ưu tiên hoặc xe đi từ các hướng khác tới.\n\n✅ **Kết luận:** Xe mô tô phải dừng lại nhường đường, **Xe con** được quyền đi trước."
    },
    {
        "question": "Câu 8: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh7.png", 
        "options": [
            "A. Xe con (A), xe cứu thương, xe con (B).", 
            "B. Xe cứu thương, xe con (B), xe con (A).", 
            "C. Xe con (B), xe con (A), xe cứu thương."
        ],
        "answer": "A",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Nhất chớm (Xe đã vào giao lộ):** Quan sát kỹ sẽ thấy **Xe con (A)** đã đi qua vạch dừng người đi bộ và tiến hẳn vào trong ngã tư. Theo luật, xe đã vào giao lộ được quyền đi trước tiên, **kể cả khi có xe ưu tiên** đi tới.\n2. **Nhì ưu (Xe ưu tiên):** Sau khi xe con (A) đi, quyền ưu tiên tiếp theo thuộc về **xe cứu thương**.\n3. **Cuối cùng:** Xe con (B) đi cuối.\n\n✅ **Thứ tự đúng:** Xe con (A) -> Xe cứu thương -> Xe con (B)."
    },
    {
        "question": "Câu 9: Thứ tự các xe đi như thế nào là đúng quy tắc giao thông?",
        "image": "images/sahinh8.png", 
        "options": [
            "A. Xe cứu thương, xe cứu hỏa, xe con.", 
            "B. Xe cứu hỏa, xe cứu thương, xe con.", 
            "C. Xe cứu thương, xe con, xe cứu hỏa."
        ],
        "answer": "B",
        "explanation": "💡 **Phân tích sa hình:**\n\n1. **Quy tắc xe ưu tiên:** Theo Luật Giao thông đường bộ, thứ tự xe ưu tiên được quy định như sau: Xe chữa cháy (cứu hỏa) $\\rightarrow$ Xe quân sự, xe công an $\\rightarrow$ Xe cứu thương.\n2. **Áp dụng:** Trong hình có 2 xe ưu tiên, đối chiếu theo luật thì xe cứu hỏa có quyền ưu tiên cao nhất, sau đó mới đến xe cứu thương. Xe con là xe bình thường đi cuối cùng.\n\n✅ **Thứ tự đúng:** Xe cứu hỏa -> Xe cứu thương -> Xe con."
    },

    # (Giữ nguyên các câu còn lại của ông)
]

# ----------------- 4. UI -----------------
st.set_page_config(page_title="Trợ lý Giao thông AI", page_icon="🚦")
st.title("🚦 Trợ lý Luật Giao thông Việt Nam")

tab1, tab2 = st.tabs(["💬 Hỏi đáp", "📝 Trắc nghiệm"])

# ================= TAB 1 =================
with tab1:
    st.header("💬 Chat & Nhận diện hình ảnh")
    st.write("Gõ câu hỏi về luật hoặc tải ảnh biển báo/tình huống lên để AI phân tích.")

    # 1. Nút tải ảnh ngay trong khung Chat
    uploaded_file = st.file_uploader("🖼️ Đính kèm ảnh (nếu có)", type=["jpg", "jpeg", "png"])
    
    # Hiển thị trước ảnh nếu có tải lên
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Ảnh đính kèm", width=250)

    # 2. Khởi tạo lịch sử Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # 3. Khung nhập câu hỏi
    if prompt := st.chat_input("Nhập câu hỏi (VD: Biển báo này có ý nghĩa gì?)..."):
        # Hiển thị câu hỏi của người dùng
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Xử lý câu trả lời của AI
        with st.chat_message("assistant"):
            with st.spinner("Đang suy nghĩ..."):
                if uploaded_file is not None:
                    # 📷 TRƯỜNG HỢP 1: CÓ ẢNH (Dùng tính năng đa phương thức của Gemini)
                    import PIL.Image
                    img = PIL.Image.open(uploaded_file)
                    
                    full_prompt = f"Bạn là chuyên gia Luật Giao thông Việt Nam. Hãy quan sát ảnh này và trả lời câu hỏi: {prompt}"
                    
                    try:
                        res = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[full_prompt, img]
                        )
                        answer = res.text
                    except Exception as e:
                        answer = f"⚠️ Lỗi khi phân tích ảnh: {e}"
                        
                else:
                    # 📝 TRƯỜNG HỢP 2: CHỈ CÓ CHỮ (Dùng RAG tra PDF như cũ)
                    docs = retriever.invoke(prompt)
                    context = "\n".join([doc.page_content for doc in docs])

                    full_prompt = f"""
Dựa trên luật sau:
{context}

Hãy trả lời câu hỏi:
{prompt}
"""
                    answer = call_gemini(full_prompt)

            # In câu trả lời ra màn hình
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# ================= TAB 2 =================
with tab2:
    st.header("📝 Trắc nghiệm")

    if 'quiz_index' not in st.session_state:
        st.session_state.quiz_index = 0
        st.session_state.score = 0
        st.session_state.submitted = False

    index = st.session_state.quiz_index

    if index < len(quiz_data):
        item = quiz_data[index]

        # 1. In câu hỏi
        st.subheader(item["question"])

        # 2. HIỂN THỊ ẢNH NẾU CÓ (Đoạn này lúc nãy ông thiếu nè)
        if "image" in item and item["image"] is not None:
            try:
                st.image(item["image"], use_container_width=True)
            except FileNotFoundError:
                st.warning(f"⚠️ Lỗi: Không tìm thấy ảnh '{item['image']}'. Ông kiểm tra lại xem lưu đúng thư mục images chưa nhé!")

        # 3. Chọn đáp án
        answer = st.radio(
            "Chọn đáp án:",
            item["options"],
            index=None,
            key=f"q_{index}"
        )

        if st.button("Nộp"):
            if answer:
                st.session_state.submitted = True

                correct = item["answer"]
                selected = answer[0]

                if selected == correct:
                    st.success("✅ Đúng!")
                    st.session_state.score += 1
                else:
                    st.error(f"❌ Sai! Đáp án đúng: {correct}")

                # ✅ KHÔNG GỌI AI → dùng sẵn explanation
                with st.expander("Giải thích"):
                    st.markdown(item["explanation"]) # Đổi thành st.markdown để hiện được chữ in đậm và emoji
            else:
                st.warning("Chọn đáp án đi!")

        if st.session_state.submitted:
            if st.button("Câu tiếp ➡️"):
                st.session_state.quiz_index += 1
                st.session_state.submitted = False
                st.rerun()

    else:
        st.balloons()
        st.success(f"Điểm: {st.session_state.score}/{len(quiz_data)}")

        if st.button("Làm lại"):
            st.session_state.quiz_index = 0
            st.session_state.score = 0
            st.session_state.submitted = False
            st.rerun()
