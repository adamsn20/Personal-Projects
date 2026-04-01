import streamlit as st
import subprocess
import os
import tempfile
from fpdf import FPDF

st.set_page_config(page_title="Code to PDF Converter", page_icon="📄")

st.title("Python & Jupyter to PDF Converter")
st.write("Upload a `.py` or `.ipynb` file to convert it into a formatted PDF.")

uploaded_file = st.file_uploader("Choose a file", type=["py", "ipynb"])

if uploaded_file is not None:
    # Determine the file type
    file_ext = uploaded_file.name.split('.')[-1]
    
    # Use a temporary directory to avoid cluttering the server
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_input_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save the uploaded file to the temp directory
        with open(temp_input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        output_pdf_name = uploaded_file.name.replace(f".{file_ext}", ".pdf")
        output_pdf_path = os.path.join(temp_dir, output_pdf_name)
        
        # --- Handle Python Files ---
        if file_ext == "py":
            with st.spinner("Converting Python script to PDF..."):
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Courier", size=10)
                
                # Read the python file and write it to the PDF
                with open(temp_input_path, "r", encoding="utf-8") as f:
                    for line in f:
                        # Clean up text encoding to prevent FPDF errors
                        clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                        pdf.cell(0, 5, txt=clean_line, ln=True)
                        
                pdf.output(output_pdf_path)
            
        # --- Handle Jupyter Notebooks ---
        elif file_ext == "ipynb":
            with st.spinner("Compiling notebook to PDF (this may take a moment)..."):
                try:
                    # Run the jupyter nbconvert command line tool
                    subprocess.run([
                        "jupyter", "nbconvert", 
                        "--to", "pdf", 
                        temp_input_path, 
                        "--output-dir", temp_dir,
                        "--output", output_pdf_name.replace('.pdf', '')
                    ], check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    st.error("Conversion failed. There might be an issue with the notebook's syntax or missing system dependencies.")
                    st.error(f"Error Log: {e.stderr}")
                    st.stop()
        
        # --- Display Download Button ---
        if os.path.exists(output_pdf_path):
            st.success("Conversion successful!")
            with open(output_pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF",
                    data=f,
                    file_name=output_pdf_name,
                    mime="application/pdf"
                )
