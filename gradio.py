from model import VoiceFakeDetection

import gradio as gr

def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("## Computer Vision Model Configuration and Training for AudioFake Detection")
        model = VoiceFakeDetection()

        with gr.Row():
            architecture = gr.Radio(
                [key for key in model.architectures.keys()],
                label="Choose the architecture"
            )
            transform = gr.Radio(
                [key for key in model.transforms.keys()],
                label="Choose the transformation for the images",
            )
            with gr.Column():
                num_epochs = gr.Number(
                    label="Number of Epochs",
                    precision=0,
                )
                num_batches = gr.Number(
                    label="Number of Batches",
                    precision=0,
                )

        callbacks = gr.Textbox(
            label="Edit your Callback",
            value="EarlyStoppingCallback(monitor='f1_score', min_delta=0.0001, patience=10)"
        )
        zipFile = gr.File(
            file_count="single",
            file_types=[".zip"]
        )
        

        output = gr.Textbox(label="Training Log")

        train_button = gr.Button("Train")

        
        train_button.click(
            fn=model.train_model,
            inputs=[architecture, transform, zipFile, num_epochs, num_batches, callbacks],
            outputs=output
        )

    return demo
    
gradio_interface().launch(share=True)
