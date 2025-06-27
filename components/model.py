from fastai.vision.all import *
import streamlit as st
import os
from context.userContext import getUserContext

class VoiceFakeDetection:
    def __init__(self) -> None:
        self.architectures = {
            'VGG16': vgg16,
            'VGG19': vgg19,
            'ResNet18': resnet18,
            'ResNet34': resnet34,
            'ResNet50': resnet50,
            'alexnet' : alexnet,
        }
        self.transforms = {
            'Resize': Resize((128, 128)),
            'Random Crop': RandomCrop(128),
        }

        # sys.stdout = StreamlitLogger()
        getUserContext() # reload user context between sessions(refresh)
        self.save_path = f"database/{st.session_state.username}"
        os.makedirs(self.save_path, exist_ok=True)


    def train_model(self, user_model_name, architecture_name, transform_type, selected_speakers, selected_noises, num_epochs, num_batches, callbacks) -> None:
        """ This method is used to train the model.
        It is called by the Streamlit app when the user clicks the 'Train' button.
        """
        self.__init_logs()

        self.user_model_name = None if user_model_name.strip()=="" else user_model_name.strip()

        if architecture_name not in self.architectures:
            st.warning("Architecture not suported.")

        if transform_type in self.transforms:
            transform = self.transforms[transform_type]
        else:
            st.warning("Transformation not suported.")

        try:
            dataset_path = "/dataset"
            tmp_paths=[f"{dataset_path}/{noise}/{spk}" for spk in selected_speakers for noise in selected_noises]

            dls = ImageDataLoaders.from_path_func(
                path=".",
                fnames=[f for path in tmp_paths for f in get_image_files(Path(path))],
                label_func=label_func,
                bs=num_batches,
                valid_pct=0.3,
                item_tfms=transform
            )
            st.session_state.dataset_info.info(f"✅ Selected {len(dls.train.dataset)+len(dls.valid.dataset)} images. Training will start soon!")
            time.sleep(1)

        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

        try:
            if self.user_model_name:
                self.model_path = f"{self.save_path}/{self.user_model_name}"
            else:
                self.model_path = f"{self.save_path}/model_{architecture_name}_{transform_type}"
            os.makedirs(self.model_path, exist_ok=True)
            self.model = vision_learner(dls, self.architectures[architecture_name], metrics=F1Score(average='macro'), path=self.model_path)
            all_callbacks = [
                CSVLogger,
                TrainingLogCallback,
                GraphCallback,
            ]
            all_callbacks.extend(callbacks)
            self.model.fine_tune(num_epochs, cbs=all_callbacks)

            self.__save_model()

            self.__losses_table()

            self.__plot_results()

            
        except Exception as e:
            st.error(f"Error in training: {str(e)}")

    
    def background_training(self, user_model_name, architecture_name, transform_type, selected_speakers, selected_noises, num_epochs, num_batches, callbacks) -> None:
        """
        This method is used to train the model in the background.
        It is called by the Streamlit app when the user clicks the 'Save Version' button.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        self.user_model_name = None if user_model_name.strip()=="" else user_model_name.strip()

        if architecture_name not in self.architectures:
            return

        if transform_type in self.transforms:
            transform = self.transforms[transform_type]
        else:
            return

        try:
            dataset_path = "/dataset"
            tmp_paths=[f"{dataset_path}/{noise}/{spk}" for spk in selected_speakers for noise in selected_noises]

            dls = ImageDataLoaders.from_path_func(
                path=".",
                fnames=[f for path in tmp_paths for f in get_image_files(Path(path))],
                label_func=label_func,
                bs=num_batches,
                valid_pct=0.3,
                item_tfms=transform
            )

        except Exception:
            return

        try:
            if self.user_model_name:
                self.model_path = f"{self.save_path}/{self.user_model_name}"
            else:
                self.model_path = f"{self.save_path}/model_{architecture_name}_{transform_type}"
            os.makedirs(self.model_path, exist_ok=True)
            self.model = vision_learner(dls, self.architectures[architecture_name], metrics=F1Score(average='macro'), path=self.model_path)
            all_callbacks = [
                CSVLogger,
            ]
            all_callbacks.extend(callbacks)
            self.model.fine_tune(num_epochs, cbs=all_callbacks)


            self.model.export("model.pkl")  

            fig, ax = plt.subplots()
            self.model.recorder.plot_loss(ax=ax)
            plt.savefig(f"{self.model_path}/results.png", bbox_inches="tight", format='png')

            interp = ClassificationInterpretation.from_learner(self.model)
            img_buffer = io.BytesIO()
            interp.plot_confusion_matrix(figsize=(7, 7), normalize=True)
            plt.savefig(f"{self.model_path}/confusion_matrix.png", format='png')

            plt.savefig(img_buffer, format='png')

            
        except Exception as e:
            return

    
    def safe_eval_callback(self, callbacks: list) -> list:
        safe_callbacks = []
        for cb in callbacks:
            cb = cb.strip()
            if cb == "":
                continue

            cb = eval(cb)
            
            # Verifica se é uma instância de Callback
            if not isinstance(cb, Callback):
                raise Exception(f"'{cb}' is not a valid fastai Callback.")

            safe_callbacks.append(cb)

        return safe_callbacks
    
    def __init_logs(self):
        st.session_state.dataset_info = st.empty()
        st.session_state.training_out = st.empty()
        st.session_state.progress =  st.progress(0.0)
        st.session_state.graph = st.empty()
    
    def __empty_logs(self):
        st.session_state.dataset_info.empty()
        st.session_state.training_out.empty()
        st.session_state.progress.empty()
        st.session_state.graph.empty()

    def __plot_results(self):

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📉 Training & Validation Loss")
            fig, ax = plt.subplots()
            self.model.recorder.plot_loss(ax=ax)
            plt.savefig(f"{self.model_path}/results.png", bbox_inches="tight", format='png')
            st.pyplot(fig, use_container_width=False)
            

        with col2:
            st.subheader("🔢 Confusion Matrix")
            interp = ClassificationInterpretation.from_learner(self.model)
            img_buffer = io.BytesIO()
            interp.plot_confusion_matrix(figsize=(7, 7), normalize=True)
            plt.savefig(f"{self.model_path}/confusion_matrix.png", format='png')

            plt.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            st.image(img_buffer, use_container_width=True)
    
    def __save_model(self):
        self.model.export("model.pkl")
        st.session_state.trained_model = f"{self.model_path}/model.pkl"

        st.success("✅ Training completed! Model saved in cache. Please download it before finishing your session.")
        st.info("All logs and results will be saved in the same folder as the model. You can download them in your Profile Page.")
        self.__empty_logs()

        model_buffer = io.BytesIO()
        with open(f"{self.model_path}/model.pkl", "rb") as file:
            model_buffer.write(file.read())
        model_buffer.seek(0)
        st.download_button(
            label="📥 Download Model",
            data=model_buffer,
            file_name=f"model.pkl" if self.user_model_name is None else self.user_model_name + ".pkl",
            mime="application/octet-stream"
        )
    
    def __losses_table(self):
        with st.expander("📊 Training Statistics", expanded=False):
            self.history_data = pd.read_csv(f"{self.model_path}/history.csv")
            st.table(self.history_data)

            csv_buffer =  io.BytesIO()
            self.history_data.to_csv(csv_buffer, index=False)

def label_func(f): 
    return f.parent.name

class TrainingLogCallback(Callback):
    def after_batch(self):
        if self.training:
            batch = self.iter
            loss = self.loss.item()
            st.session_state.dataset_info.empty()
            st.session_state.training_out.code(f"Epoch {self.epoch}: \nBatch {batch}: Loss {loss:.4f}")
            st.session_state.progress.progress(batch/ self.n_iter)

    def after_epoch(self):
        st.session_state.progress.progress(1.0)
        st.session_state.training_out.code(f"Epoch {self.epoch} complete!")

class GraphCallback(Callback):
    def after_loss(self):
        fig, ax = plt.subplots(figsize=(4, 4))
        self.recorder.plot_loss(ax=ax, show_epochs=True)
        ax.set_ylim(0, max(1, ax.get_ylim()[1]))
        plt.tight_layout()
        st.session_state.graph.pyplot(fig, use_container_width=False)
        plt.close(fig)
