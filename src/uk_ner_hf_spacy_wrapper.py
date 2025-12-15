import os
import spacy
import torch  # Не забудьте імпортувати torch
from spacy.language import Language
from spacy.tokens import Span
from transformers import pipeline

@Language.factory("uk_ner_pure_dl")
class UKNerPureComponent:
    def __init__(self, nlp, name, finetuned_path: str):
        device_num = 0 if torch.cuda.is_available() else -1
        
        print(f"Loading HF model from: {finetuned_path} on device {device_num}...")

        # Завантажуємо навчену модель з Hugging Face
        self.hf_pipeline = pipeline(
            "ner",
            model=finetuned_path,
            tokenizer=finetuned_path,
            aggregation_strategy="simple",
            device=device_num
        )

    def __call__(self, doc):
        try:
            hf_ents = self.hf_pipeline(doc.text)
        except Exception as e:
            print(f"Inference error: {e}")
            hf_ents = []

        spacy_ents = []
        seen_tokens = set()

        for ent in hf_ents:
            # 2. Мапимо координати символів на токени spaCy
            span = doc.char_span(ent['start'], ent['end'], alignment_mode="expand")

            if span:
                # Уникаємо перетинів сутностей
                if any(t.i in seen_tokens for t in span):
                    continue

                label = ent['entity_group']

                try:
                    spacy_ents.append(Span(doc, span.start, span.end, label=label))
                    seen_tokens.update(t.i for t in span)
                except Exception:
                    pass

        # 3. Фільтруємо і зберігаємо
        spacy_ents = sorted(spacy_ents, key=lambda x: x.start)
        doc.ents = spacy.util.filter_spans(spacy_ents)
        return doc

def build_spacy_wrapper(output_path: str, finetuned_path: str, component_name: str = "uk_ner_pure_dl"):
    """
    Створює порожню модель spaCy та додає до неї DL компонент, передаючи шлях через конфіг.
    """
    
    if not os.path.exists(output_path):
        print(f"Creating model directory: {output_path}...")
        
        nlp_build = spacy.blank("uk")

        # ВИПРАВЛЕННЯ: Перевіряємо наявність у глобальному реєстрі spaCy
        if component_name in spacy.registry.factories:
             print(f"Adding component '{component_name}' to the pipeline...")
             
             # Передаємо finetuned_path через config
             nlp_build.add_pipe(
                 component_name, 
                 config={"finetuned_path": finetuned_path}, 
                 last=True
             )
             
             # Зберігаємо на диск
             nlp_build.to_disk(output_path)
             print(f"Model wrapper saved successfully to {output_path}\n")
        else:
            print(f"ERROR: Component factory '{component_name}' not found in registry.") 
            print("Available factories:", list(spacy.registry.factories.get_all().keys()))
    else:
        print(f"Directory {output_path} already exists. Skipping creation.")