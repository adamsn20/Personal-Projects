import streamlit as st
import subprocess
import os
import tempfile
from fpdf import FPDF

st.set_page_config(page_title="Batch Code to PDF Converter", page_icon="📄")

st.title("Batch Python & Jupyter to PDF Converter")
st.write("Upload multiple `.py` or `.ipynb` files to convert them into formatted PDFs.")

# Enable multiple file uploads
uploaded_files = st.file_uploader("Choose files", type=["py", "ipynb"], accept_multiple_files=True)

# Check if the list of uploaded files is not empty
if uploaded_files:
    
    # Initialize the progress bar and a placeholder for status text
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create a single temporary directory for the entire batch
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Loop through the list of uploaded files
        for i, uploaded_file in enumerate(uploaded_files):
            
            # Update the status text so the user knows which file is processing
            status_text.text(f"Processing {uploaded_file.name} ({i+1} of {len(uploaded_files)})...")
            
            file_ext = uploaded_file.name.split('.')[-1]
            temp_input_path = os.path.join(temp_dir, uploaded_file.name)
            
            # Save the current uploaded file
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
                    continue # Skip to the next file in the loop if this one fails
            
            # --- Display Individual Download Button ---
            if os.path.exists(output_pdf_path):
                with open(output_pdf_path, "rb") as f:
                    st.download_button(
                        label=f"Download {output_pdf_name}",
                        data=f,
                        file_name=output_pdf_name,
                        mime="application/pdf",
                        key=f"download_{i}" # The unique key prevents Streamlit duplicate widget errors
                    )
            
            # Update the progress bar dynamically
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        # Final success message once the loop finishes
        status_text.success("All files processed successfully!")
