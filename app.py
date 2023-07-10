import streamlit as st
from PIL import Image
from clf import predict

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("\n\n\nPadiKita [Prototype Prediction]")
st.write("")

file_up = st.file_uploader("Upload an image", type=["jpg", "png"])

if file_up is not None:
    image = Image.open(file_up)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Just a second...")
    label, skor = predict(file_up)

    st.write("Prediction classes: ", label)
    st.write(skor)
