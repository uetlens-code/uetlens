from uetlens.intervener import Intervener

iv = Intervener(
    model_path ="/path/to/Meta-Llama-3.1-8B",
    sae_path = "/path/to/sae/OpenSAE-LLaMA-3.1-Layer_00",
    use_multi_gpu=True
)

iv.run_intervention_experiment(
    input_prompt = '''Event Type Classification Task: Classify whether the marked event is an attack event.

    Event types: [attack, none]

    Examples:
    Sentence: "<event>As Kienmayer's columns fled to the east, they joined with elements of the Russian Empire's army in a rear guard action at the Battle of Amstetten on 5 November.</event>" Event type: none
    Sentence: "<event>On 15 June, the Pakistani military intensified air strikes in North Waziristan and bombed eight foreign militant hideouts.</event>" Event type: attack
    Sentence: "<event>This had trapped the French army in Egypt on the African side of the Mediterranean, and all efforts to reinforce and resupply them had ended in failure.</event>" Event type: none
    Sentence: "<event>The Hindus later attacked a Muslim dargah, and Muslim protesters also attacked the temple again, leading to a mass breakout of violence.</event>" Event type: attack

    Now analyze this sentence:
    Sentence: "<event>Utah, along with Sword on the eastern flank, was added to the invasion plan in December 1943.</event>" Event type:''',
    intervention_indices=[101989],
    output_path="out/manipulate/event_attack_00.txt",
    num_generations=10,
    max_new_tokens=1,
    temperature=0.7,
    experiment_name="event_attack_L0"
)