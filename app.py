#!/usr/bin/env python
# filepath: /Users/tristanhearn/code/maximum-submatrix-sum/app.py

import gradio as gr
import numpy as np
import pandas as pd
from algorithms import brute_submatrix_max, fft_submatrix_max, kidane_max_submatrix

# Function to generate a random 10x10 matrix with numbers having 1 decimal place
def generate_random_matrix(rows=10, cols=10):
    # Generate random floats between -10 and 10 with 1 decimal place
    matrix = np.round(np.random.uniform(-10, 10, size=(rows, cols)), 1)
    # Convert to Pandas DataFrame
    df = pd.DataFrame(matrix)
    return df

# Function to process the matrix and find the maximum submatrix sum
def process_matrix(matrix_df, algorithm):
    try:
        # Convert input to numpy array
        matrix_array = matrix_df.values.astype(float)
        
        # Select the appropriate algorithm
        if algorithm == "Brute Force":
            loc, max_sum, time_taken = brute_submatrix_max(matrix_array)
        elif algorithm == "FFT":
            loc, max_sum, time_taken = fft_submatrix_max(matrix_array)
        else:  # Kidane method
            loc, max_sum, time_taken = kidane_max_submatrix(matrix_array)
        print(f"Using algorithm: {algorithm}")
        
        # Format the result message
        result_message = (
            f"Algorithm used: {algorithm}\n"
            f"Maximum Submatrix Sum: {max_sum:.2f}\n"
            f"Time taken: {time_taken:.6f} seconds\n"
            f"Submatrix location: Rows {loc[0].start} to {loc[0].stop-1}, Columns {loc[1].start} to {loc[1].stop-1}"
        )
        
        # Create a styled DataFrame with highlighted cells
        df = pd.DataFrame(matrix_array)
        
        # Create a mask for the maximum submatrix
        mask = pd.DataFrame(np.zeros_like(matrix_array, dtype=bool))
        mask.iloc[loc[0], loc[1]] = True
        
        # Apply background color based on the mask
        def highlight_max_submatrix(val):
            color = 'background-color: #90EE90'  # Light green
            default = ''
            return np.where(mask, color, default)
        
        # Style the DataFrame with the highlighting
        styled_df = df.style.apply(highlight_max_submatrix, axis=None)
        
        # Attempt to render styled DataFrame to HTML using to_html, fallback on string conversion if necessary
        try:
            html_output = styled_df.to_html()
        except Exception as e:
            html_output = str(styled_df)
        return html_output, result_message
    except Exception as e:
        print(f"Error in process_matrix: {e}")
        return matrix_df, str(e)

# Initialize Gradio interface
with gr.Blocks(title="Maximum Submatrix Sum Calculator") as app:
    gr.Markdown("# Maximum Submatrix Sum Calculator")
    gr.Markdown("Edit the matrix below or use the random generator, then select an algorithm to find the maximum sum submatrix.")
    
    random_matrix_btn = gr.Button("Generate New Random Matrix")
    
    # Use a dataframe component for the matrix input/output
    matrix_display = gr.Dataframe(
        value=generate_random_matrix(),
        interactive=True,
        label="Matrix (cells in max submatrix will be highlighted in green)"
    )
    
    highlighted_matrix = gr.HTML(label="Highlighted Matrix")
    
    with gr.Row():
        algorithm = gr.Radio(
        ["Brute Force", "FFT", "Kidane"], 
        value="FFT",
            label="Algorithm"
        )
    
    with gr.Row():
        submit_btn = gr.Button("Find Maximum Submatrix Sum", variant="primary")
    
    result_text = gr.Textbox(label="Results", lines=3)
    
    # Event handlers
    random_matrix_btn.click(generate_random_matrix, outputs=[matrix_display])
    submit_btn.click(
        process_matrix, 
        inputs=[matrix_display, algorithm], 
        outputs=[highlighted_matrix, result_text]
    )

    # Print a message before launching
    print("Launching Gradio app for Maximum Submatrix Sum Calculator...")

# Run the app
if __name__ == "__main__":
    app.launch(show_error=True)
