import streamlit as st
from PIL import Image
import requests

url = "http://localhost:8000/make_model"
# response = requests.get(url)
# data = response.json()

# 가상의 학습 함수
def click(file_path):
    # 여기에서 실제 학습 로직을 넣으면 됩니다.
    # 학습 후에 결과 이미지를 반환
    print(file_path[-4]+".png")
    result_image = Image.open(file_path[-4]+".png")
    return result_image

def main():
    st.title("학습 애플리케이션")

    # 1. 텍스트 박스에 파일 경로 입력
    file_path = st.text_input("파일 경로 입력", "/path/to/your/file.csv")

    # 2. 학습 버튼
    if st.button("학습 시작"):
        if not file_path:
            st.warning("파일 경로를 입력하세요.")
        else:
            # 학습 시뮬레이션 함수 호출
            data = {"path": file_path} 
            response = requests.post(url, json=data, timeout=30)
            result_image = Image.open("power_consumption_data_with_volatility.png")
            st.success("학습이 완료되었습니다!")

            # 3. 학습 후 결과 이미지 표시
            st.image(result_image, caption="학습 결과", use_column_width=True)

if __name__ == "__main__":
    main()
