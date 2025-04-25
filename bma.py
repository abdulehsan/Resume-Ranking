from rank_bm25 import BM25Okapi

def applybm25(df_processed_text,query):
    
    if isinstance(df_processed_text, list) and isinstance(df_processed_text[0], list):
        bm25 = BM25Okapi(df_processed_text)
        
    
        #query = ['achyuth','adelina','erimia','adhi','gopalam','software','design','development']
        
        
        scores = bm25.get_scores(query)
        return scores
    else:
        raise ValueError("Input is not in the correct format. It should be a list of lists.")
