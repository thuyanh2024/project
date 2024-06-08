import streamlit as st
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pretrained model
model = load_model("best_model.h5")

# Define class names
classname = {1: "Ung thư", 0: "Bình thường"}


def processed_img(img):
    img = img.resize((224, 224))  # Resize the image
    img_array = img_to_array(img)
    img_array = img_array.astype('float32')
    img_array /= 255.0
    img_array = img_array.reshape((1, 224, 224, 3))  # Reshape for model input
    output = model.predict(img_array)[0]
    y_class = output.argmax()
    result = classname[y_class]
    return result


def main():
    # Add custom CSS for a light green color gradient background and styling the prediction result
    st.markdown(
        """
        <style>
        body {
            background: linear-gradient(to right, #e0eafc, #cfdef3);
        }
        .main {
            background: linear-gradient(to right, #e0eafc, #cfdef3);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
        }
        h1, h3 {
            color: #333;
        }
        .result {
            background-color: #ffcccb;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            color: #b30000;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='main'>", unsafe_allow_html=True)
    st.title("Phần mềm AI nhận diện khối u ác tính từ hình ảnh tổn thương da")
    st.markdown("### Trần Ngọc Thùy Anh - Phạm Việt Thành")

    img_file = st.file_uploader("Tải lên một hình ảnh:", type=["jpg", "png"])

    if img_file is not None:
        # Display the uploaded image
        uploaded_img = Image.open(img_file)
        st.image(uploaded_img, caption='Hình ảnh đã tải lên', use_column_width=True)

        # Process the image and get the predicted disease
        disease_prediction = processed_img(uploaded_img)
        
        # Display the predicted disease
        if disease_prediction == "Ung thư":
            st.markdown(f"<div class='result'>Kết quả dự đoán: {disease_prediction}</div>", unsafe_allow_html=True)
        else:
            st.success(f"Kết quả dự đoán: {disease_prediction}")

        # Display additional information based on the predicted disease
        if disease_prediction == "Bệnh nấm móng":
            st.info("Thông tin thêm về Bệnh nấm móng sẽ hiển thị tại đây.")
        elif disease_prediction == "Bệnh trứng cá đỏ":
            st.info("Thông tin thêm về Bệnh trứng cá đỏ sẽ hiển thị tại đây.")
        elif disease_prediction == "Bệnh ung thư hắc tố":
            st.info("Thông tin thêm về Bệnh ung thư hắc tố sẽ hiển thị tại đây.")
    
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
