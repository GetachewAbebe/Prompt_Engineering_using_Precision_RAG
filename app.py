import os
import json
import streamlit as st
from utils.data_generation import main as generate_prompts
from utils.retrieval import retrieve_context
from utils.evaluation import main1 as evaluate_accuracy
from utils.ranking import evaluate_prompt

# Constants
TEST_DATA_PATH = os.path.join("test_dataset", "test_data.json")
TEST_OUTPUT_COUNT = 5

# --- Set up Streamlit page ---
st.set_page_config(page_title="Prompt Generation App", page_icon="🤖", layout="wide")

# --- Utility Functions ---
@st.cache_data
def load_json(file_path):
    """Load a JSON file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return {}

@st.cache_data
def generate_test_data(query):
    """Generate prompts and retrieve test data."""
    generate_prompts(str(TEST_OUTPUT_COUNT), query)
    return load_json(TEST_DATA_PATH)


@st.cache_data
def evaluate_test_prompts():
    """Evaluate the accuracy of generated prompts."""
    return evaluate_accuracy()


def render_table(data, column_names):
    """Render data in a table format."""
    formatted_data = {col: [row[i] for row in data] for i, col in enumerate(column_names)}
    st.table(formatted_data)


def handle_file_upload(uploaded_file):
    """Validate and handle uploaded files."""
    if uploaded_file:
        try:
            file_content = uploaded_file.read()
            st.success(f"File {uploaded_file.name} uploaded successfully.")
            return file_content
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            return None
    return None


# --- Sidebar Functions ---
def render_sidebar(search_history_list):
    """Render the sidebar with search history and file uploader."""
    with st.sidebar.expander("Search History"):
        st.text_area("History:", height=200, value="\n".join(search_history_list))

    return st.sidebar.file_uploader("Upload a File", type=["txt", "json", "pdf"])


# --- Main Functionalities ---
def generate_prompts_section(query_input):
    """Generate prompts based on user input."""
    if not query_input:
        st.error("Please enter a query before generating prompts.")
        return

    with st.spinner("Generating prompts..."):
        try:
            test_data = generate_test_data(query_input)
            questions = [item["user"] for item in test_data]
            st.write("Generated Questions:")
            st.write(questions)
        except Exception as e:
            st.error(f"An error occurred during prompt generation: {e}")


def evaluate_prompts_section():
    """Evaluate prompts for accuracy and display results."""
    try:
        test_data = load_json(TEST_DATA_PATH)
        questions = [item["user"] for item in test_data]
        accuracies = evaluate_test_prompts()

        # Sort by accuracy
        sorted_data = sorted(zip(questions, accuracies), key=lambda x: x[1], reverse=True)
        ranks = [i + 1 for i in range(len(sorted_data))]

        st.write("Evaluation Results:")
        render_table(sorted_data, ["Rank", "Questions", "Accuracy"])
    except Exception as e:
        st.error(f"An error occurred during prompt evaluation: {e}")


def rank_prompts_section(query_input):
    """Rank prompts using Monte Carlo and Elo rating systems."""
    try:
        test_data = load_json(TEST_DATA_PATH)
        questions = [item["user"] for item in test_data]

        evaluations = evaluate_prompt(query_input, questions)
        data = [
            {
                "Prompt": q,
                "Monte Carlo Score": evaluations[f'test_case_{i+1}']['Monte Carlo Evaluation'],
                "Elo Rating": evaluations[f'test_case_{i+1}']['Elo Rating Evaluation']
            }
            for i, q in enumerate(questions)
        ]
        st.write("Ranked Prompts (Elo and Monte Carlo):")
        st.dataframe(data)
    except Exception as e:
        st.error(f"An error occurred during prompt ranking: {e}")


# --- Main App Layout ---
def main():
    # Render Title
    st.title("Prompt Generation App 🤖")

    # Initialize search history
    search_history_list = []

    # Query Input
    query_input = st.text_input("Enter your query:", value="")

    # Sidebar
    uploaded_file = render_sidebar(search_history_list)
    handle_file_upload(uploaded_file)

    # Buttons for actions
    col1, col2, col3 = st.columns(3)
    with col1:
        generate_button = st.button("Generate Prompts")
    with col2:
        evaluate_button = st.button("Evaluate Prompts")
    with col3:
        rank_button = st.button("Rank Prompts")

    # Generate Prompts Section
    if generate_button:
        generate_prompts_section(query_input)

    # Evaluate Prompts Section
    if evaluate_button:
        evaluate_prompts_section()

    # Rank Prompts Section
    if rank_button:
        rank_prompts_section(query_input)


if __name__ == "__main__":
    main()
