import streamlit as st
from PIL import Image
import requests

url = "http://localhost:8000/make_model"
# response = requests.get(url)
# data = response.json()

def click(file_path):
    result_image = Image.open(file_path[-4]+".png")
    return result_image

def main():
    st.title("AutoML Training")

    file_path = st.text_input("data", "/path/to/your/file.csv")
    max_trials = st.text_input("max trials:", "50")
    epochs = st.text_input("epochs:", "20")
    if st.button("학습 시작"):
        if not file_path:
            st.warning("파일 경로를 입력하세요.")
        else:
            data = {"path": file_path, "max_trials": max_trials, "epochs": epochs} 
            response = requests.post(url, json=data, timeout=300)
            result_image = Image.open(f"{file_path[:-4]}.png")
            st.success("학습이 완료되었습니다!")

            st.image(result_image, caption="학습 결과", use_column_width=True)

if __name__ == "__main__":
    main()
