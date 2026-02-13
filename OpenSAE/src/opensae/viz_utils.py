import json
import numpy as np
import os
from matplotlib import cm

def colorize_token(token, activation_value, max_activation_value):
    if max_activation_value == 0:
        normalized_activation = 0
    else:
        normalized_activation = np.clip(activation_value / max_activation_value, 0, 1)

    if normalized_activation == 0:
        return f'<span style="color:#888;">{token}</span>'
    elif normalized_activation < 0.3:
        return f'<span style="color:#FFB6C1;">{token}</span>'
    elif normalized_activation < 0.7:
        return f'<span style="color:#FF6347;">{token}</span>'
    else:
        return f'<span style="color:#FF0000; font-weight:bold;">{token}</span>'

def generate_html(activations, base_vector_indices, frc_scores=None):
    if frc_scores is not None:
        sorted_vectors = []
        for base_vec in base_vector_indices:
            frc = frc_scores.get(base_vec, 0)
            sorted_vectors.append((base_vec, frc))
        
        sorted_vectors.sort(key=lambda x: x[1], reverse=True)
        base_vector_indices = [vec for vec, frc in sorted_vectors]
        
        sort_info = " (Sorted by FRC)"
    else:
        sort_info = ""

    html_content = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: "Microsoft YaHei", "Arial", sans-serif;
                background-color: #f4f4f4;
                display: flex;
                justify-content: center;
                padding: 20px;
            }}
            .container {{
                display: flex;
                width: 80%;
            }}
            .sidebar {{
                width: 25%;
                margin-right: 20px;
                background-color: #fff;
                padding: 10px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                position: sticky;
                top: 20px;
                height: 80vh;
                overflow-y: auto;
            }}
            .vector-list {{
                display: flex;
                flex-direction: column;
                gap: 10px;
            }}
            .vector-item {{
                padding: 10px;
                background-color: #f0f0f0;
                border-radius: 5px;
                cursor: pointer;
                text-align: left;
                flex-shrink: 0;
                border-left: 4px solid #007acc;
            }}
            .vector-item:hover {{
                background-color: #ddd;
            }}
            .vector-item.frc-high {{
                border-left-color: #28a745;
                background-color: #f8fff9;
            }}
            .vector-item.frc-medium {{
                border-left-color: #ffc107;
                background-color: #fffef0;
            }}
            .vector-item.frc-low {{
                border-left-color: #dc3545;
                background-color: #fff5f5;
            }}
            .frc-score {{
                float: right;
                font-weight: bold;
                color: #666;
            }}
            .content {{
                width: 75%;
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                max-height: 80vh;
                overflow-y: auto;
            }}
            .sentence {{
                margin-bottom: 20px;
                font-size: 16px;
                line-height: 1.6;
            }}
            .sentence h4 {{
                margin-bottom: 10px;
                color: #333;
            }}
            h2, h3 {{
                color: #333;
            }}
            strong {{
                color: #333;
            }}
            .sort-info {{
                color: #666;
                font-size: 14px;
                margin-bottom: 10px;
            }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-datalabels"></script>
        <script>
            function showVectorContent(baseVectorIndex) {{
                var contentSections = document.querySelectorAll('.vector-content');
                contentSections.forEach(function(section) {{
                    section.style.display = 'none';
                }});
                var selectedContent = document.getElementById('vector-' + baseVectorIndex);
                if (selectedContent) {{
                    selectedContent.style.display = 'block';
                }}
            }}

            document.addEventListener('DOMContentLoaded', function() {{
                var firstVector = document.querySelector('.vector-item');
                if (firstVector) {{
                    var firstVectorIndex = firstVector.getAttribute('onclick').match(/(\\d+)/)[0];
                    showVectorContent(firstVectorIndex);
                }}
            }});
        </script>
    </head>
    <body>
    <div class="container">
        <div class="sidebar">
            <h3>Base Vectors{sort_info}</h3>
            <div class="vector-list" id="vector-list">
    """

    for base_vector_index in base_vector_indices:
        frc_score = frc_scores.get(base_vector_index, 0) if frc_scores else 0

        if frc_score >= 0.8:
            frc_class = "frc-high"
        elif frc_score >= 0.5:
            frc_class = "frc-medium"
        else:
            frc_class = "frc-low"
            
        html_content += f'<div class="vector-item {frc_class}" onclick="showVectorContent({base_vector_index})">'
        html_content += f'Base Vector {base_vector_index}'
        if frc_scores:
            html_content += f'<span class="frc-score">{frc_score:.3f}</span>'
        html_content += '</div>'

    html_content += """
            </div>
        </div>

        <div class="content">
            <h2>Activation Visualization</h2>
    """

    for base_vector_index in base_vector_indices:
        frc_score = frc_scores.get(base_vector_index, 0) if frc_scores else 0
        
        html_content += f'<div class="vector-content" id="vector-{base_vector_index}" style="display:none;">'
        html_content += f"<h3>Base Vector {base_vector_index}"
        if frc_scores:
            html_content += f" <span style='color:#666; font-size:16px;'>(FRC: {frc_score:.3f})</span>"
        html_content += "</h3>"
        
        sentence_counter = 0
        for sentence_data in activations:
            sentence_counter += 1
            sentence_id = sentence_data.get("sentence_id", "Unknown")
            tokens = sentence_data.get("tokens", [])
            
            colorized_sentence = f'<div class="sentence"><strong>Sentence ID: {sentence_id}</strong><br>'

            token_list = []
            activation_list = []

            for token_data in tokens:
                token = token_data.get("token", "")
                token_list.append(token)
                activation = 0
                for top in token_data.get("activations", []):
                    if top.get("base_vector") == base_vector_index:
                        activation = top.get("activation", 0)
                        break
                activation_list.append(activation)
                colorized_token = colorize_token(token, activation, max(max(activation_list) if activation_list else 1, 1))
                colorized_sentence += colorized_token + " "

            colorized_sentence += '</div>'
            html_content += colorized_sentence

            canvas_id = f"chart-{base_vector_index}-{sentence_counter}"
            html_content += f'<canvas id="{canvas_id}" width="400" height="200"></canvas>'

            tokens_js = json.dumps(token_list, ensure_ascii=False)
            activation_js = json.dumps(activation_list)

            html_content += f"""
            <script>
            (function() {{
                var ctx = document.getElementById("{canvas_id}").getContext("2d");
                new Chart(ctx, {{
                    type: 'bar',
                    data: {{
                        labels: {tokens_js},
                        datasets: [{{
                            label: 'Activation Value',
                            data: {activation_js},
                            backgroundColor: 'rgba(54, 162, 235, 0.6)',
                            borderColor: 'rgba(54, 162, 235, 1)',
                            borderWidth: 1
                        }}]
                    }},
                    options: {{
                        plugins: {{
                            datalabels: {{
                                anchor: 'end',
                                align: 'end',
                                formatter: function(value) {{
                                    return value === 0 ? '' : value.toFixed(3);
                                }},
                                font: {{
                                    weight: 'bold'
                                }}
                            }}
                        }},
                        scales: {{
                            y: {{
                                beginAtZero: true
                            }}
                        }}
                    }},
                    plugins: [ChartDataLabels]
                }});
            }})();
            </script>
            """

        html_content += "</div>"

    html_content += """
        </div>
    </div>
    </body>
    </html>
    """
    return html_content

def save_html_file(html_content, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)