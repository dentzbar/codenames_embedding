import streamlit as st
import random
import numpy as np
from utils import load_words

# Color mappings for the game
COLOR_MAP = {
    0: "#808080",  # Grey (innocent)
    1: "#FF4444",  # Red (red team)
    2: "#4444FF"   # Blue (blue team)
}

COLOR_NAMES = {
    0: "Grey",
    1: "Red", 
    2: "Blue"
}

def get_word_list():
    """
    Placeholder function to fetch 25 words for the game grid.
    Returns a list of 25 words.
    """
    # Try to load words from the saved dataset
    words = load_words('simple_word_dataset.pkl')
    
    if words is None:
        # Fallback to a simple word list if no saved dataset
        fallback_words = [
            "apple", "book", "car", "dog", "elephant", "fire", "guitar", "house", "ice", "jungle",
            "key", "lion", "moon", "night", "ocean", "piano", "queen", "river", "sun", "tree",
            "umbrella", "violin", "water", "xray", "yellow", "zebra", "mountain", "bridge", "castle", "dream"
        ]
        words = fallback_words
    
    # Select 25 random words
    selected_words = random.sample(words, min(25, len(words)))
    return selected_words

def get_color_map():
    """
    Placeholder function to generate color mapping for the 25 word grid.
    Returns a list of 25 integers (0=grey/innocent, 1=red team, 2=blue team).
    """
    # Typical Codenames distribution: 9 red, 8 blue, 7 innocent, 1 assassin (using grey for assassin)
    colors = [1] * 9 + [2] * 8 + [0] * 8  # 9 red, 8 blue, 8 grey/innocent
    random.shuffle(colors)
    return colors

def play_game(selected_words):
    """
    Placeholder function that simulates gameplay.
    Returns a list of words that were "guessed" this turn.
    """
    # For demo purposes, randomly select 1-3 words from the selected words
    num_guesses = random.randint(1, min(3, len(selected_words)))
    guessed_words = random.sample(selected_words, num_guesses)
    return guessed_words

def initialize_game():
    """Initialize a new game with words and color mapping."""
    words = get_word_list()
    colors = get_color_map()
    
    # Create 5x5 grids
    word_grid = np.array(words).reshape(5, 5)
    color_grid = np.array(colors).reshape(5, 5)
    
    return word_grid, color_grid

def check_game_over(revealed_grid, color_grid):
    """Check if the game is over (one color is fully revealed)."""
    for color in [1, 2]:  # Red and Blue teams
        color_positions = (color_grid == color)
        if np.all(revealed_grid[color_positions]):
            return True, COLOR_NAMES[color]
    return False, None

def main():
    st.set_page_config(page_title="Codenames Game", layout="wide")
    st.title("🎯 Codenames Game")
    
    # Initialize session state
    if 'game_initialized' not in st.session_state:
        st.session_state.game_initialized = False
        st.session_state.game_over = False
        st.session_state.winner = None
    
    # New Game button
    if st.button("🎮 New Game", type="primary"):
        word_grid, color_grid = initialize_game()
        st.session_state.word_grid = word_grid
        st.session_state.color_grid = color_grid
        st.session_state.revealed_grid = np.zeros((5, 5), dtype=bool)
        st.session_state.game_initialized = True
        st.session_state.game_over = False
        st.session_state.winner = None
        st.rerun()
    
    if not st.session_state.game_initialized:
        st.info("👆 Click 'New Game' to start playing!")
        return
    
    # Game layout
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader("🎯 Game Board")
        
        # Display the 5x5 word grid
        for i in range(5):
            cols = st.columns(5)
            for j in range(5):
                with cols[j]:
                    word = st.session_state.word_grid[i, j]
                    color = st.session_state.color_grid[i, j]
                    is_revealed = st.session_state.revealed_grid[i, j]
                    
                    if is_revealed:
                        # Show the word with its team color
                        st.markdown(
                            f"""<div style='
                                background-color: {COLOR_MAP[color]};
                                color: white;
                                padding: 20px;
                                text-align: center;
                                border-radius: 10px;
                                font-weight: bold;
                                margin: 2px;
                                height: 60px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                            '>{word.upper()}</div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        # Show unrevealed word
                        st.markdown(
                            f"""<div style='
                                background-color: #f0f0f0;
                                color: black;
                                padding: 20px;
                                text-align: center;
                                border-radius: 10px;
                                font-weight: bold;
                                margin: 2px;
                                height: 60px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                border: 2px solid #ddd;
                            '>{word.upper()}</div>""",
                            unsafe_allow_html=True
                        )
    
    with col2:
        st.subheader("🎮 Controls")
        
        if not st.session_state.game_over:
            if st.button("▶️ Play Turn", type="secondary"):
                # Get unrevealed words
                unrevealed_words = []
                for i in range(5):
                    for j in range(5):
                        if not st.session_state.revealed_grid[i, j]:
                            unrevealed_words.append(st.session_state.word_grid[i, j])
                
                if unrevealed_words:
                    # Simulate gameplay
                    guessed_words = play_game(unrevealed_words)
                    
                    # Reveal the guessed words
                    for word in guessed_words:
                        for i in range(5):
                            for j in range(5):
                                if st.session_state.word_grid[i, j] == word:
                                    st.session_state.revealed_grid[i, j] = True
                    
                    # Check for game over
                    game_over, winner = check_game_over(
                        st.session_state.revealed_grid, 
                        st.session_state.color_grid
                    )
                    
                    if game_over:
                        st.session_state.game_over = True
                        st.session_state.winner = winner
                    
                    st.rerun()
        else:
            st.success(f"🎉 Game Over! {st.session_state.winner} team wins!")
    
    with col3:
        st.subheader("🗺️ Spymaster Map")
        st.caption("(This shows the secret color mapping)")
        
        # Display the color map
        for i in range(5):
            cols = st.columns(5)
            for j in range(5):
                with cols[j]:
                    color = st.session_state.color_grid[i, j]
                    word = st.session_state.word_grid[i, j]
                    is_revealed = st.session_state.revealed_grid[i, j]
                    
                    # Add border for revealed words
                    border_style = "3px solid #000" if is_revealed else "1px solid #333"
                    
                    st.markdown(
                        f"""<div style='
                            background-color: {COLOR_MAP[color]};
                            color: white;
                            padding: 15px;
                            text-align: center;
                            border-radius: 8px;
                            font-weight: bold;
                            margin: 2px;
                            height: 50px;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            border: {border_style};
                            font-size: 12px;
                        '>{word[:8].upper()}</div>""",
                        unsafe_allow_html=True
                    )
    
    # Game statistics
    if st.session_state.game_initialized:
        st.subheader("📊 Game Statistics")
        
        # Count revealed words by color
        red_total = np.sum(st.session_state.color_grid == 1)
        blue_total = np.sum(st.session_state.color_grid == 2)
        grey_total = np.sum(st.session_state.color_grid == 0)
        
        red_revealed = np.sum((st.session_state.color_grid == 1) & st.session_state.revealed_grid)
        blue_revealed = np.sum((st.session_state.color_grid == 2) & st.session_state.revealed_grid)
        grey_revealed = np.sum((st.session_state.color_grid == 0) & st.session_state.revealed_grid)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🔴 Red Team", 
                value=f"{red_revealed}/{red_total}",
                delta=f"{red_total - red_revealed} remaining"
            )
        
        with col2:
            st.metric(
                label="🔵 Blue Team", 
                value=f"{blue_revealed}/{blue_total}",
                delta=f"{blue_total - blue_revealed} remaining"
            )
        
        with col3:
            st.metric(
                label="⚪ Innocent", 
                value=f"{grey_revealed}/{grey_total}",
                delta=f"{grey_total - grey_revealed} remaining"
            )

if __name__ == "__main__":
    main() 