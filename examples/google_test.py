import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

_ = load_dotenv()
from litewebagent.core.agent_factory import setup_function_calling_web_agent
from litewebagent.webagent_utils_sync.utils.playwright_manager import setup_playwright

agent_type = "FunctionCallingAgent"
starting_url = "https://www.google.com"
# goal = '숭실대학교의 학과를 검색하고 단과대별 학과를 csv 파일로 정리해서 저장해줘.'
goal = '12월에 롯데월드 근처에서 할만한 데이트 코스를 검색하고, 블로그 게시물 등을 참고해서 csv 파일에 행선지를 정리해서 저장해줘.'
plan = None
log_folder = "log"
# model = "gpt-4o-mini"
model = "gemini-2.5-flash"
features = ["axtree"]
branching_factor = None
elements_filter = "som"
storage_state = None

playwright_manager = setup_playwright(storage_state='state.json', headless=False)
agent = setup_function_calling_web_agent(
    starting_url,
    goal,
    playwright_manager=playwright_manager,
    model_name=model,
    agent_type=agent_type,
    features=features,
    elements_filter=elements_filter,
    tool_names=["navigation", "select_option", "upload_file", "webscraping", "save_file"],
    branching_factor=branching_factor,
    log_folder=log_folder
)
response = agent.send_prompt(plan)
print(response)
