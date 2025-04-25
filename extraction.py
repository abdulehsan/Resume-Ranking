import docx2txt as dr
import pandas as pd
from io import BytesIO
# List to store extracted data
data = []

def extract_text(uploaded_files):
    global df  
    data.clear() 

    
    for uploaded_file in uploaded_files:
        
        file_data = BytesIO(uploaded_file.read())

        
        if uploaded_file.name.lower().endswith(".docx"):
            text = dr.process(file_data)
            data.append({
                "File Name": uploaded_file.name,
                "Resume Text": text
            })

    
    df = pd.DataFrame(data)
    return df
