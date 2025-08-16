#!/usr/bin/env python3
"""
AI vs AI Codenames Game using Semantic Code Word Analysis

Two AI players with different embedding models compete:
- Red Player: Uses one embedding model (e.g., BERT)
- Blue Player: Uses another embedding model (e.g., OpenAI)

Game Rules:
- 25 words total: 9 red team, 8 blue team, 8 innocents
- Blue team starts first
- Each turn, the current player uses semantic analysis to generate a clue
- The opponent/innocents are treated as "avoid" words in the analysis
- Players automatically reveal words based on similarity rankings
"""

import os
import random
from typing import List, Dict, Set, Tuple
import time

from utils import load_words
from semantic_codeword_analyzer import SemanticCodewordAnalyzer


# ANSI Colors for display
ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[34m"
ANSI_RED = "\033[31m"
ANSI_GRAY = "\033[90m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BOLD = "\033[1m"
ANSI_BOLD_BLUE = "\033[1;34m"
ANSI_BOLD_RED = "\033[1;31m"
ANSI_BOLD_GREEN = "\033[1;32m"
ANSI_MAGENTA = "\033[35m"


class AICodenamesGame:
    """AI vs AI Codenames game with semantic code word analysis"""
    
    def __init__(
        self, 
        red_vectorstore_path: str = "data/wordnet_bert_embedded_store.pkl",
        blue_vectorstore_path: str = "data/wordnet_openai_embedded_store.pkl",
        word_dataset_path: str = "data/simple_word_dataset.pkl",
        seed: int = None,
        verbose: bool = True,
        enable_sleep: bool = True
    ):
        self.red_vectorstore_path = red_vectorstore_path
        self.blue_vectorstore_path = blue_vectorstore_path
        self.word_dataset_path = word_dataset_path
        self.seed = seed
        self.verbose = verbose
        self.enable_sleep = enable_sleep
        
        # Game state
        self.red_agent_words: List[str] = []
        self.blue_agent_words: List[str] = []
        self.innocent_words: List[str] = []
        self.all_words: List[str] = []
        
        self.revealed_words: Set[str] = set()
        self.current_team = "red"  # blue starts first
        self.game_over = False
        self.winner = None
        
        # Game statistics
        self.turn_count = 0
        self.red_remaining = 9
        self.blue_remaining = 8
        
        # AI Players
        self.red_player = None
        self.blue_player = None
        
    def setup_game(self) -> None:
        """Initialize the game board and AI players"""
        if self.seed is not None:
            random.seed(self.seed)
            
        # Load word dataset
        game_words = load_words(self.word_dataset_path)
        
        # Sample 25 words total
        selected_words = [w.lower() for w in random.sample(game_words, 25)]
        
        # Assign words to teams
        self.red_agent_words = selected_words[:9]
        self.blue_agent_words = selected_words[9:17] 
        self.innocent_words = selected_words[17:25]
        self.all_words = selected_words.copy()
        random.shuffle(self.all_words)
        
        print(f"{ANSI_BOLD_GREEN}🎮 AI vs AI CODENAMES GAME 🎮{ANSI_RESET}")
        print(f"Red team: {self.red_remaining} words")
        print(f"Blue team: {self.blue_remaining} words") 
        print(f"Innocents: {len(self.innocent_words)} words")
        print()
        
        # Initialize AI players
        print(f"{ANSI_BOLD_RED}🤖 Red Player: {os.path.basename(self.red_vectorstore_path)}{ANSI_RESET}")
        print(f"{ANSI_BOLD_BLUE}🤖 Blue Player: {os.path.basename(self.blue_vectorstore_path)}{ANSI_RESET}")
        
    def display_board(self, show_colors: bool = False) -> None:
        """Display the current game board"""
        print(f"\n{ANSI_BOLD}📋 GAME BOARD{ANSI_RESET}")
        print("=" * 70)
        
        # Display in 5x5 grid
        for i in range(5):
            row = []
            for j in range(5):
                word_idx = i * 5 + j
                word = self.all_words[word_idx]
                
                if word in self.revealed_words:
                    # Revealed words - show with team colors
                    if word in self.red_agent_words:
                        row.append(f"{ANSI_RED}[{word.upper()}]{ANSI_RESET}")
                    elif word in self.blue_agent_words:
                        row.append(f"{ANSI_BLUE}[{word.upper()}]{ANSI_RESET}")
                    else:
                        row.append(f"{ANSI_GRAY}[{word.upper()}]{ANSI_RESET}")
                elif show_colors:
                    # Show actual colors (for debugging)
                    if word in self.red_agent_words:
                        row.append(f"{ANSI_RED}{word}{ANSI_RESET}")
                    elif word in self.blue_agent_words:
                        row.append(f"{ANSI_BLUE}{word}{ANSI_RESET}")
                    else:
                        row.append(f"{ANSI_GRAY}{word}{ANSI_RESET}")
                else:
                    # Hidden words
                    row.append(word)
            
            print(" | ".join(f"{w:12}" for w in row))
        
        print("=" * 70)
        print(f"Red remaining: {ANSI_BOLD_RED}{self.red_remaining}{ANSI_RESET} | "
              f"Blue remaining: {ANSI_BOLD_BLUE}{self.blue_remaining}{ANSI_RESET}")
        
    def get_remaining_team_words(self, team: str) -> List[str]:
        """Get unrevealed words for the specified team"""
        if team == "red":
            return [w for w in self.red_agent_words if w not in self.revealed_words]
        else:
            return [w for w in self.blue_agent_words if w not in self.revealed_words]
    
    def get_opponent_and_innocent_words(self, team: str) -> List[str]:
        """Get all non-team words (opponents + innocents) that are unrevealed"""
        if team == "red":
            opponent_words = [w for w in self.blue_agent_words if w not in self.revealed_words]
        else:
            opponent_words = [w for w in self.red_agent_words if w not in self.revealed_words]
            
        innocent_words = [w for w in self.innocent_words if w not in self.revealed_words]
        return opponent_words + innocent_words
    
    def run_encoder_analysis(self, team: str) -> Tuple[int, str]:
        """Run encoder analysis for the current team and return optimal_n and best_word"""
        remaining_team_words = self.get_remaining_team_words(team)
        opponent_innocent_words = self.get_opponent_and_innocent_words(team)
        
        team_color = ANSI_BOLD_RED if team == "red" else ANSI_BOLD_BLUE
        vectorstore_path = self.red_vectorstore_path if team == "red" else self.blue_vectorstore_path
        
        print(f"\n{team_color}🧠 {team.upper()} PLAYER THINKING...{ANSI_RESET}")
        print(f"Model: {os.path.basename(vectorstore_path)}")
        print(f"Target words: {len(remaining_team_words)}")
        print(f"Avoid words: {len(opponent_innocent_words)}")
        
        # Create a custom analyzer for this turn
        class GameAnalyzer(SemanticCodewordAnalyzer):
            def __init__(self, team_words, avoid_words, vectorstore):
                super().__init__(vectorestore_path=vectorstore)
                self.custom_team_words = team_words
                self.custom_avoid_words = avoid_words
                
            def _sample_board(self):
                # Override to use current game state
                self.game_agents = self.custom_team_words.copy()
                self.game_innocents = self.custom_avoid_words.copy()
                
            def _print_board(self):
                # Custom board display for game context
                print(f"{team_color}Team words: {', '.join(self.custom_team_words[:3])}{'...' if len(self.custom_team_words) > 3 else ''}{ANSI_RESET}")
                print(f"{ANSI_GRAY}Avoid words: {', '.join(self.custom_avoid_words[:3])}{'...' if len(self.custom_avoid_words) > 3 else ''}{ANSI_RESET}")
        
        # Run analysis
        analyzer = GameAnalyzer(remaining_team_words, opponent_innocent_words, vectorstore_path)
        result = analyzer.run()
        
        return result.optimal_n, result.best_word
    
    def run_decoder_analysis(self, best_word: str, optimal_n: int) -> List[Tuple[str, float]]:
        """Run decoder analysis using all board words and return similarity rankings"""
        # Use the current team's vectorstore for decoding
        vectorstore_path = self.red_vectorstore_path if self.current_team == "red" else self.blue_vectorstore_path
        
        print(f"\n{ANSI_MAGENTA}🔍 DECODER ANALYSIS{ANSI_RESET}")
        print(f"Analyzing clue: '{best_word}' for {optimal_n} words")
        
        # Create analyzer just for decoding
        analyzer = SemanticCodewordAnalyzer(vectorestore_path=vectorstore_path)
        analyzer._load_vectorestore()
        
        # Get all unrevealed words for comparison
        comparison_words = [w for w in self.all_words if w not in self.revealed_words]
        
        # Run decoder analysis (without printing, we'll handle display in game)
        similarities = analyzer._decoder_analysis(best_word, comparison_words, print_results=False)
        
        return similarities
    
    def ai_guessing_phase(self, best_word: str, optimal_n: int, similarities: List[Tuple[str, float]]) -> None:
        """AI guessing phase - always ends after all guesses are made"""
        team_color = ANSI_BOLD_RED if self.current_team == "red" else ANSI_BOLD_BLUE
        
        print(f"\n{team_color}🎯 {self.current_team.upper()} TEAM CLUE: '{best_word.upper()}' for {optimal_n} word(s){ANSI_RESET}")
        
        # AI gets optimal_n
        num_guesses = optimal_n
        guesses_made = 0
        
        print(f"AI will make up to {num_guesses} guesses based on similarity rankings:")
        
        while guesses_made < num_guesses and not self.game_over:
            if guesses_made >= len(similarities):
                print(f"No more words to guess!")
                break
                
            # AI picks the highest similarity unrevealed word
            guess, similarity = similarities[guesses_made]
            guesses_made += 1
            
            print(f"\nGuess {guesses_made}/{num_guesses}: '{guess}' (similarity: {similarity:.3f})")
            
            # Check team membership BEFORE revealing the word
            is_current_team = guess in self.get_remaining_team_words(self.current_team)
            is_opponent_team = guess in self.get_remaining_team_words("blue" if self.current_team == "red" else "red")
            
            # Now reveal the word
            self.revealed_words.add(guess)
            
            if is_current_team:
                # Correct guess!
                if self.current_team == "red":
                    self.red_remaining -= 1
                    print(f"{ANSI_BOLD_GREEN}✅ Correct! '{guess}' is a RED word.{ANSI_RESET}")
                else:
                    self.blue_remaining -= 1
                    print(f"{ANSI_BOLD_GREEN}✅ Correct! '{guess}' is a BLUE word.{ANSI_RESET}")
                
                # Check win condition
                if (self.current_team == "red" and self.red_remaining == 0) or \
                   (self.current_team == "blue" and self.blue_remaining == 0):
                    self.game_over = True
                    self.winner = self.current_team
                    
            else:
                # Wrong guess - turn ends
                if is_opponent_team:
                    opponent = "blue" if self.current_team == "red" else "red"
                    if opponent == "red":
                        self.red_remaining -= 1
                    else:
                        self.blue_remaining -= 1
                    print(f"{ANSI_BOLD_RED}❌ Wrong! '{guess}' belongs to the {opponent.upper()} team.{ANSI_RESET}")
                else:
                    print(f"{ANSI_BOLD_RED}❌ Wrong! '{guess}' is an innocent word.{ANSI_RESET}")
                
                # Check if opponent won by accident
                if (self.current_team == "red" and self.blue_remaining == 0) or \
                   (self.current_team == "blue" and self.red_remaining == 0):
                    self.game_over = True
                    self.winner = "blue" if self.current_team == "red" else "red"
                
                # Wrong guess ends the turn early
                print(f"{ANSI_YELLOW}Turn ends due to wrong guess.{ANSI_RESET}")
                # Show updated board
                self.display_board()
                if self.enable_sleep:
                    time.sleep(1)
                break
            
            # Show updated board
            self.display_board()
            if self.enable_sleep:
                time.sleep(1)  # Pause for readability
        
        print(f"{ANSI_YELLOW}End of {self.current_team.upper()} team's turn.{ANSI_RESET}")
    
    def play_turn(self) -> None:
        """Play a single turn"""
        self.turn_count += 1
        team_color = ANSI_BOLD_RED if self.current_team == "red" else ANSI_BOLD_BLUE
        
        print(f"\n{'='*80}")
        print(f"{team_color}🎯 TURN {self.turn_count} - {self.current_team.upper()} TEAM{ANSI_RESET}")
        print(f"{'='*80}")
        
        remaining_words = self.get_remaining_team_words(self.current_team)
        if len(remaining_words) == 0:
            self.game_over = True
            self.winner = self.current_team
            return
        
        # 1. Encoder phase: Generate clue
        optimal_n, best_word = self.run_encoder_analysis(self.current_team)
        
        # 2. Decoder phase: Analyze similarities
        similarities = self.run_decoder_analysis(best_word, optimal_n)
        
        # 3. Guessing phase: AI makes guesses
        self.ai_guessing_phase(best_word, optimal_n, similarities)
        
        # Always switch teams after each turn (no matter the outcome)
        if not self.game_over:
            self.current_team = "blue" if self.current_team == "red" else "red"
    
    def play_game(self) -> None:
        """Main game loop (interactive mode)"""
        self.setup_game()
        
        print(f"\n{ANSI_BOLD_GREEN}🎮 GAME STARTING! 🎮{ANSI_RESET}")
        print("Blue team (AI) goes first!")
        
        # Show initial board
        print(f"\n{ANSI_YELLOW}Initial board (colors hidden):{ANSI_RESET}")
        self.display_board()
        
        # Main game loop
        while not self.game_over and self.turn_count < 20:  # Safety limit
            input(f"\n{ANSI_YELLOW}Press Enter to start {self.current_team} team's turn...{ANSI_RESET}")
            self.play_turn()
        
        # Game over
        print(f"\n{'='*80}")
        print(f"{ANSI_BOLD_GREEN}🎉 GAME OVER! 🎉{ANSI_RESET}")
        print(f"{'='*80}")
        
        if self.winner:
            winner_color = ANSI_BOLD_RED if self.winner == "red" else ANSI_BOLD_BLUE
            print(f"{winner_color}{self.winner.upper()} TEAM WINS!{ANSI_RESET}")
            
            # Show which model won
            winning_model = self.red_vectorstore_path if self.winner == "red" else self.blue_vectorstore_path
            print(f"Winning model: {os.path.basename(winning_model)}")
        else:
            print("Game ended in a draw or reached turn limit!")
            
        print(f"Total turns: {self.turn_count}")
        
        # Show final board with all colors
        print(f"\n{ANSI_YELLOW}Final board (all colors revealed):{ANSI_RESET}")
        self.display_board(show_colors=True)
    
    def play_game_batch(self) -> Dict:
        """Main game loop (batch mode - no user interaction)"""
        import time
        start_time = time.time()
        
        # Temporarily disable verbose mode for batch processing
        original_verbose = self.verbose
        self.verbose = False
        
        self.setup_game()
        
        # Main game loop (no user interaction)
        while not self.game_over and self.turn_count < 20:  # Safety limit
            self.play_turn()
        
        # Restore verbose mode
        self.verbose = original_verbose
        
        game_duration = time.time() - start_time
        
        # Return results for batch processing
        return {
            'winner': self.winner if self.winner else 'draw',
            'red_score': 9 - self.red_remaining,  # Words found
            'blue_score': 8 - self.blue_remaining,  # Words found
            'total_turns': self.turn_count,
            'game_duration': game_duration
        }


def main():
    """Main function to start the AI vs AI game"""
    print(f"{ANSI_BOLD_GREEN}Welcome to AI vs AI Semantic Codenames!{ANSI_RESET}")
    print("Two AI players with different embedding models will compete.")
    print()
    
    # Game options
    use_seed = input("Use random seed for reproducible games? (y/n): ").lower().startswith('y')
    seed = 42 if use_seed else None
    
    # Model selection - all 6 pairwise combinations
    print("\nAvailable model combinations:")
    print("1. BERT vs OpenAI (default)")
    print("2. BERT vs Gemini") 
    print("3. BERT vs RoBERTa")
    print("4. OpenAI vs Gemini")
    print("5. OpenAI vs RoBERTa")
    print("6. Gemini vs RoBERTa")
    print("7. Custom paths")
    
    choice = input("Choose option (1-7, or Enter for default): ").strip()
    
    if choice == "2":
        red_model = "data/wordnet_bert_embedded_store.pkl"
        blue_model = "data/wordnet_gemini_embedded_store.pkl"
    elif choice == "3":
        red_model = "data/wordnet_bert_embedded_store.pkl"
        blue_model = "data/wordnet_roberta_embedded_store.pkl"
    elif choice == "4":
        red_model = "data/wordnet_openai_embedded_store.pkl"
        blue_model = "data/wordnet_gemini_embedded_store.pkl"
    elif choice == "5":
        red_model = "data/wordnet_openai_embedded_store.pkl"
        blue_model = "data/wordnet_roberta_embedded_store.pkl"
    elif choice == "6":
        red_model = "data/wordnet_gemini_embedded_store.pkl"
        blue_model = "data/wordnet_roberta_embedded_store.pkl"
    elif choice == "7":
        red_model = input("Red team model path: ").strip()
        blue_model = input("Blue team model path: ").strip()
    else:  # Default (option 1)
        red_model = "data/wordnet_bert_embedded_store.pkl"
        blue_model = "data/wordnet_openai_embedded_store.pkl"
    
    # Create and start game
    game = AICodenamesGame(
        red_vectorstore_path=red_model,
        blue_vectorstore_path=blue_model,
        seed=seed
    )
    game.play_game()


if __name__ == "__main__":
    main() 