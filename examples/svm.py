from uetlens.svm import EventTypeSVMClassifier
import os

model_path = "/path/to/Meta-Llama-3.1-8B"
sae_path_template = "/path/to/sae/OpenSAE-LLaMA-3.1-Layer_{:02d}"

classifier = EventTypeSVMClassifier(
    model_path=model_path,
    sae_path_template=sae_path_template,
    cuda_devices="4,5,6,7"
)

training_data = [
    ('<event>The army launched an attack on the rebel stronghold</event>', 'Attack'),
    ('<event>Militants bombed the government building yesterday</event>', 'Attack'),
    ('<event>The truck transported goods across the border</event>', 'Transport'),
    ('<event>Five people died in the car accident</event>', 'Die'),
    ('<event>Several civilians were injured in the explosion</event>', 'Injure'),
    ('<event>Leaders from both countries met to discuss peace</event>', 'Meet'),
    ('<event>The committee elected a new chairman</event>', 'Elect'),
    ('<event>The court held a hearing on the case</event>', 'Trial')
]

classifier.train_svm(training_data, output_dir="out/svm_event_type")

test_sentences = [
    '<event>Rebels attacked the military base at dawn</event>',
    '<event>The convoy moved through the desert</event>',
    '<event>Three people lost their lives in the fire</event>',
    '<event>Many were wounded in the explosion</event>',
    '<event>Diplomats met to negotiate the treaty</event>',
    '<event>The committee elected new leadership</event>',
    '<event>The court hearing lasted all day</event>'
]

true_labels = ['Attack', 'Transport', 'Die', 'Injure', 'Meet', 'Elect', 'Trial']

predictions, probabilities = classifier.predict(test_sentences, true_labels, output_dir="out/svm_event_type")
print(f"predictions: {predictions}")