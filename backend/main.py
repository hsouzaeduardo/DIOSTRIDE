import os
import base64
from openai import AzureOpenAI
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
from dotenv import load_dotenv
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Caminho absoluto até o arquivo .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

endpoint = os.getenv("ENDPOINT_URL")
deployment = os.getenv("DEPLOYMENT_NAME")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
print("AZURE_OPENAI_API_KEY:", os.getenv("AZURE_OPENAI_API_KEY"))

# Inicializar o cliente OpenAI do Azure com autenticação baseada em chave
client = AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=subscription_key,
    api_version="2025-01-01-preview",
)

app = FastAPI()

# Adiciona CORS para todos os endereços
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-image/")
async def analyze_image(
    file: UploadFile = File(...),
    tipo_aplicacao: str = Form(...),
    autenticacao: str = Form(...),
    acesso_internet: str = Form(...),
    dados_sensiveis: str = Form(...),
    descricao_aplicacao: str = Form(...)
):
    try:
        # Salvar arquivo temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Converter imagem para base64
        with open(tmp_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("ascii")

        # Montar prompt para o modelo
        prompt_text = criar_prompt_modelo_ameacas(
            tipo_aplicacao,
            autenticacao,
            acesso_internet,
            dados_sensiveis,
            descricao_aplicacao
        )
        chat_prompt = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Você é um assistente de IA que ajuda as pessoas a encontrar informações."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "descreve a imagem por gentielza"
                    }
                ]
            }
        ]

        # Chamar o modelo
        completion = client.chat.completions.create(
            model=deployment,
            messages=chat_prompt,
            max_tokens=800,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )
        # Limpar arquivo temporário
        os.remove(tmp_path)
        return JSONResponse(content=completion.to_dict())
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

def criar_prompt_modelo_ameacas(tipo_aplicacao, autenticacao, acesso_internet, dados_sensiveis, descricao_aplicacao):
    prompt = f"""
Aja como um especialista em cibersegurança com mais de 20 anos de experiência utilizando a metodologia de modelagem de ameaças STRIDE para produzir modelos de ameaças abrangentes para uma ampla gama de aplicações. Sua tarefa é analisar o resumo do código, o conteúdo do README e a descrição da aplicação fornecidos para produzir uma lista de ameaças específicas para essa aplicação.

Preste atenção especial ao conteúdo do README, pois ele frequentemente fornece contexto valioso sobre o propósito do projeto, arquitetura e possíveis considerações de segurança.

Para cada uma das categorias do STRIDE (Falsificação de Identidade - Spoofing, Violação de Integridade - Tampering, Repúdio - Repudiation, Divulgação de Informações - Information Disclosure, Negação de Serviço - Denial of Service, e Elevação de Privilégio - Elevation of Privilege), liste múltiplas (3 ou 4) ameaças credíveis, se aplicável. Cada cenário de ameaça deve apresentar uma situação plausível em que a ameaça poderia ocorrer no contexto da aplicação.

Ao fornecer o modelo de ameaças, utilize uma resposta formatada em JSON com as chaves "threat_model" e "improvement_suggestions". Em "threat_model", inclua um array de objetos com as chaves "Threat Type" (Tipo de Ameaça), "Scenario" (Cenário), e "Potential Impact" (Impacto Potencial).

Inclua também o modelo de ameaças em um fluxo de trabalho utilizando mermaidjs para ilustrar as relações entre as ameaças e os componentes da aplicação. O fluxo de trabalho deve ser formatado como um array de strings, onde cada string representa uma linha do diagrama mermaidjs.
O diagrama deve incluir os seguintes elementos:
- Cada tipo de ameaça (Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, Elevation of Privilege) deve ser representado como um nó.
- As ameaças devem ser conectadas aos componentes relevantes da aplicação, mostrando como cada ameaça se relaciona com os diferentes elementos do sistema.
- As conexões entre os nós devem ser claras e indicar a direção da ameaça (por exemplo, de um componente para outro).
O diagrama deve ser claro e fácil de entender, permitindo que os desenvolvedores visualizem rapidamente as ameaças potenciais e suas relações com os componentes da aplicação.
O diagrama deve ser formatado como um array de strings, onde cada string representa uma linha do diagrama mermaidjs.
As ameaças devem ser específicas para a aplicação em questão e não devem incluir ameaças genéricas ou irrelevantes. O modelo de ameaças deve ser claro e conciso, permitindo que os desenvolvedores compreendam rapidamente as ameaças potenciais e suas implicações.

Em "improvement_suggestions", inclua um array de strings que sugerem quais informações adicionais poderiam ser fornecidas para tornar o modelo de ameaças mais completo e preciso na próxima iteração. Foque em identificar lacunas na descrição da aplicação que, se preenchidas, permitiriam uma análise mais detalhada e precisa, como por exemplo:
- Detalhes arquiteturais ausentes que ajudariam a identificar ameaças mais específicas
- Fluxos de autenticação pouco claros que precisam de mais detalhes
- Descrição incompleta dos fluxos de dados
- Informações técnicas da stack não informadas
- Fronteiras ou zonas de confiança do sistema não especificadas
- Descrição incompleta do tratamento de dados sensíveis

Não forneça recomendações de segurança genéricas — foque apenas no que ajudaria a criar um modelo de ameaças melhor.

TIPO DE APLICAÇÃO: {tipo_aplicacao}
MÉTODOS DE AUTENTICAÇÃO: {autenticacao}
EXPOSTA NA INTERNET: {acesso_internet}
DADOS SENSÍVEIS: {dados_sensiveis}
RESUMO DE CÓDIGO, CONTEÚDO DO README E DESCRIÇÃO DA APLICAÇÃO:
{descricao_aplicacao}

Exemplo de formato esperado em JSON:

    {{
      "threat_model": [
        {{
          "Threat Type": "Spoofing",
          "Scenario": "Cenário de exemplo 1",
          "Potential Impact": "Impacto potencial de exemplo 1"
        }},
        {{
          "Threat Type": "Spoofing",
          "Scenario": "Cenário de exemplo 2",
          "Potential Impact": "Impacto potencial de exemplo 2"
        }}
        // ... mais ameaças
      ],
      "improvement_suggestions": [
        "Por favor, forneça mais detalhes sobre o fluxo de autenticação entre os componentes para permitir uma análise melhor de possíveis falhas de autenticação.",
        "Considere adicionar informações sobre como os dados sensíveis são armazenados e transmitidos para permitir uma análise mais precisa de exposição de dados.",
        // ... mais sugestões para melhorar o modelo de ameaças
      ]
    }}

    Exemplo de formato esperado em mermaidjs:
    graph TD
    Usuario[Usuário]
    EasyAuth["Easy Auth Autenticação"]
    AppService[Serviço de Aplicação]
    SQL[(Azure SQL Database)]
    Monitoramento[Application Insights / Azure Monitor]
    Identidade[Microsoft Entra ID]

    Usuario -->|Token OAuth2| EasyAuth
    EasyAuth --> AppService
    AppService --> SQL
    AppService --> Monitoramento
    EasyAuth --> Identidade

    %% Ameaças Spoofing
    Spoofing1[Spoofing:\nToken roubado] --> EasyAuth
    Spoofing2[Spoofing:\nConta falsa] --> AppService

    %% Ameaças Tampering
    Tampering1[Tampering:\nInterceptação de requisições] --> AppService
    Tampering2[Tampering:\nAlteração indevida de dados] --> SQL

    %% Ameaça Repudiation
    Repudiation1[Repudiation:\nNegação de ações] --> AppService

    %% Ameaça Information Disclosure
    InfoLeak1[Information Disclosure:\nExposição de dados] --> SQL

    %% Ameaça Denial of Service
    DoS1[Denial of Service:\nSobrecarga da aplicação] --> AppService

    %% Ameaça Elevation of Privilege
    Privilege1[Elevation of Privilege:\nEscalada de privilégios] --> AppService

    %% Estilo visual (opcional)
    classDef threat fill:#ffe6e6, stroke:#ff4d4d, stroke-width:2px;
    class Spoofing1,Spoofing2,Tampering1,Tampering2,Repudiation1,InfoLeak1,DoS1,Privilege1 threat;


"""
    return prompt