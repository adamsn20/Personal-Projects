import streamlit as st
import subprocess
import os
import tempfile
import zipfile
import io
from fpdf import FPDF

st.set_page_config(page_title="Python & Jupyter Notebook to PDF Converter", page_icon="📄")

st.title("Python & Jupyter Notebook to PDF Converter")
st.write("Upload `.py` or `.ipynb` files to convert them into formatted PDFs.")

# --- 1. Initialize Session State ---
if "pdf_outputs" not in st.session_state:
    st.session_state.pdf_outputs = {}

uploaded_files = st.file_uploader("Choose files", type=["py", "ipynb"], accept_multiple_files=True)

if uploaded_files:
    # --- 2. The Action Button ---
    if st.button("Convert Files"):
        st.session_state.pdf_outputs = {} 
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name} ({i+1} of {len(uploaded_files)})...")
                
                file_ext = uploaded_file.name.split('.')[-1]
                temp_input_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(temp_input_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    
                output_pdf_name = uploaded_file.name.replace(f".{file_ext}", ".pdf")
                output_pdf_path = os.path.join(temp_dir, output_pdf_name)
                
                # --- Handle Python Files ---
                if file_ext == "py":
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_font("Courier", size=10)
                    with open(temp_input_path, "r", encoding="utf-8") as f:
                        for line in f:
                            clean_line = line.encode('latin-1', 'replace').decode('latin-1')
                            pdf.cell(0, 5, txt=clean_line, ln=True)
                    pdf.output(output_pdf_path)
                    
                # --- Handle Jupyter Notebooks ---
                elif file_ext == "ipynb":
                    try:
                        subprocess.run([
                            "jupyter", "nbconvert", 
                            "--to", "pdf", 
                            temp_input_path, 
                            "--output-dir", temp_dir,
                            "--output", output_pdf_name.replace('.pdf', '')
                        ], check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e:
                        st.error(f"Failed to convert {uploaded_file.name}. There may be syntax errors.")
                        st.error(f"Error Log: {e.stderr}")
                        continue 
                
                # --- 3. Save to Memory ---
                if os.path.exists(output_pdf_path):
                    with open(output_pdf_path, "rb") as f:
                        st.session_state.pdf_outputs[output_pdf_name] = f.read()
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            status_text.success("All files processed successfully!")

# --- 4. Render Download Buttons ---
if st.session_state.pdf_outputs:
    st.divider()
    st.subheader("Your PDFs are ready:")
    
    # Create an in-memory ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for pdf_name, pdf_bytes in st.session_state.pdf_outputs.items():
            # Write each PDF from session state into the zip archive
            zip_file.writestr(pdf_name, pdf_bytes)
    
    # --- The "Download All" Button ---
    # We place this at the top so it's easy to find
    st.download_button(
        label="📦 Download All (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="converted_pdfs.zip",
        mime="application/zip",
        type="primary" # Makes the button stand out visually
    )
    
    st.write("Or download individually:")
    
    # --- The Individual Download Buttons ---
    for pdf_name, pdf_bytes in st.session_state.pdf_outputs.items():
        st.download_button(
            label=f"📄 Download {pdf_name}",
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            key=f"dl_{pdf_name}" 
        )
