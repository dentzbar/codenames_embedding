#!/usr/bin/env python3
"""
Batch Game Experiments - AI vs AI Codenames

Runs systematic experiments with all embedding model combinations:
- All 6 pairwise combinations of 4 models
- 20 games per combination (10 with each team assignment)
- 10 different seeds for reproducibility
- Results saved to CSV for analysis
"""

import os
import csv
import itertools
from typing import List, Dict, Tuple
import time
from datetime import datetime

from game import AICodenamesGame


# Available embedding models
MODELS = {
    "bert": "data/wordnet_bert_embedded_store.pkl",
    "openai": "data/wordnet_openai_embedded_store.pkl", 
    "gemini": "data/wordnet_gemini_embedded_store.pkl",
    "roberta": "data/wordnet_roberta_embedded_store.pkl"
}

# Seeds for reproducible experiments
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021, 2223, 2425, 2627, 2829, 3031, 3233, 3435, 3637, 3839, 4041, 4243, 4445, 4647, 4849, 5051, 5253, 5455, 5657, 5859, 6061, 6263, 6465, 6667, 6869, 7071, 7273, 7475, 7677, 7879, 8081, 8283, 8485, 8687, 8889, 9091, 9293, 9495, 9697, 9899, 10101]
print('number of seeds:', len(SEEDS))

class BatchGameExperiments:
    """Run batch experiments of AI vs AI Codenames games"""
    
    def __init__(self, output_file: str = None):
        self.output_file = output_file or f"data/game_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.results = []
        
    def run_single_game(self, red_model: str, blue_model: str, red_path: str, blue_path: str, seed: int) -> Dict:
        """Run a single game and return results"""
        try:
            game = AICodenamesGame(
                red_vectorstore_path=red_path,
                blue_vectorstore_path=blue_path,
                seed=seed,
                verbose=False  # Suppress output for batch processing
            )
            
            # Run game in batch mode (no user interaction)
            result = game.play_game_batch()
            
            return {
                'red_model': red_model,
                'blue_model': blue_model,
                'seed': seed,
                'winner': result['winner'],
                'red_score': result['red_score'],
                'blue_score': result['blue_score'],
                'total_turns': result['total_turns'],
                'game_duration': result['game_duration']
            }
            
        except Exception as e:
            print(f"Error in game {red_model} vs {blue_model} (seed {seed}): {e}")
            return {
                'red_model': red_model,
                'blue_model': blue_model,
                'seed': seed,
                'winner': 'ERROR',
                'red_score': -1,
                'blue_score': -1,
                'total_turns': -1,
                'game_duration': -1
            }
    
    def run_model_combination(self, model1: str, model2: str) -> List[Dict]:
        """Run all games for a specific model combination"""
        path1 = MODELS[model1]
        path2 = MODELS[model2]
        
        combination_results = []
        
        print(f"\n🎯 Running {model1} vs {model2} combination...")
        print(f"   Red={model1}, Blue={model2} (10 games)")
        
        # First 10 games: model1=red, model2=blue
        for i, seed in enumerate(SEEDS):
            print(f"   Game {i+1}/10: seed={seed}", end=" ")
            start_time = time.time()
            
            result = self.run_single_game(model1, model2, path1, path2, seed)
            result['game_duration'] = time.time() - start_time
            
            combination_results.append(result)
            print(f"→ {result['winner']} wins ({result['red_score']}-{result['blue_score']}) [{result['game_duration']:.1f}s]")
        
        print(f"   Red={model2}, Blue={model1} (10 games)")
        
        # Next 10 games: model2=red, model1=blue (reversed roles)
        for i, seed in enumerate(SEEDS):
            print(f"   Game {i+11}/20: seed={seed}", end=" ")
            start_time = time.time()
            
            result = self.run_single_game(model2, model1, path2, path1, seed)
            result['game_duration'] = time.time() - start_time
            
            combination_results.append(result)
            print(f"→ {result['winner']} wins ({result['red_score']}-{result['blue_score']}) [{result['game_duration']:.1f}s]")
        
        return combination_results
    
    def run_all_experiments(self):
        """Run all pairwise model combinations"""
        print("🚀 Starting Batch Game Experiments")
        print("=" * 60)
        print(f"Models: {list(MODELS.keys())}")
        print(f"Seeds: {SEEDS}")
        print(f"Output: {self.output_file}")
        print("=" * 60)
        
        # Generate all unique pairs (no self-play, no duplicates)
        model_pairs = list(itertools.combinations(MODELS.keys(), 2))
        total_combinations = len(model_pairs)
        
        print(f"Model combinations to test: {total_combinations}")
        for i, (m1, m2) in enumerate(model_pairs):
            print(f"  {i+1}. {m1} vs {m2}")
        
        start_time = time.time()
        
        # Run each combination
        for i, (model1, model2) in enumerate(model_pairs):
            print(f"\n{'='*60}")
            print(f"COMBINATION {i+1}/{total_combinations}: {model1.upper()} vs {model2.upper()}")
            print(f"{'='*60}")
            
            combination_results = self.run_model_combination(model1, model2)
            self.results.extend(combination_results)
            
            # Save intermediate results
            self.save_results()
            
            # Progress summary
            wins_by_model = {}
            for result in combination_results:
                winner = result['winner']
                if winner not in wins_by_model:
                    wins_by_model[winner] = 0
                wins_by_model[winner] += 1
            
            print(f"\n📊 Combination Summary:")
            for model, wins in wins_by_model.items():
                print(f"   {model}: {wins} wins")
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 ALL EXPERIMENTS COMPLETE!")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total games: {len(self.results)}")
        print(f"Results saved to: {self.output_file}")
        
        # Final summary
        self.print_final_summary()
    
    def save_results(self):
        """Save results to CSV file"""
        fieldnames = ['red_model', 'blue_model', 'seed', 'winner', 'red_score', 'blue_score', 'total_turns', 'game_duration']
        
        with open(self.output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
    
    def print_final_summary(self):
        """Print final experiment summary"""
        if not self.results:
            return
            
        print(f"\n📊 FINAL SUMMARY")
        print("=" * 60)
        
        # Overall win rates by model
        model_stats = {}
        for model in MODELS.keys():
            model_stats[model] = {'wins': 0, 'games': 0}
        
        for result in self.results:
            red_model = result['red_model']
            blue_model = result['blue_model']
            winner = result['winner']
            
            # Count games played
            model_stats[red_model]['games'] += 1
            model_stats[blue_model]['games'] += 1
            
            # Count wins
            if winner in model_stats:
                model_stats[winner]['wins'] += 1
        
        print("Model Performance:")
        for model, stats in model_stats.items():
            win_rate = stats['wins'] / stats['games'] * 100 if stats['games'] > 0 else 0
            print(f"  {model:8}: {stats['wins']:2}/{stats['games']:2} wins ({win_rate:5.1f}%)")
        
        # Head-to-head matrix
        print(f"\nHead-to-Head Results (wins):")
        models = list(MODELS.keys())
        
        # Create matrix
        matrix = {}
        for m1 in models:
            matrix[m1] = {}
            for m2 in models:
                matrix[m1][m2] = 0
        
        for result in self.results:
            red = result['red_model']
            blue = result['blue_model']
            winner = result['winner']
            
            if winner == red:
                matrix[red][blue] += 1
            elif winner == blue:
                matrix[blue][red] += 1
        
        # Print matrix
        print(f"{'':10}", end="")
        for model in models:
            print(f"{model:8}", end="")
        print()
        
        for m1 in models:
            print(f"{m1:10}", end="")
            for m2 in models:
                if m1 == m2:
                    print(f"{'--':>8}", end="")
                else:
                    print(f"{matrix[m1][m2]:>8}", end="")
            print()


def main():
    """Main function to run batch experiments"""
    print("🎮 Batch Game Experiments - AI vs AI Codenames")
    print("This will run systematic experiments with all model combinations.")
    print()
    
    # Check if model files exist
    missing_models = []
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            missing_models.append(f"{model_name}: {model_path}")
    
    if missing_models:
        print("❌ Missing model files:")
        for missing in missing_models:
            print(f"   {missing}")
        print("Please ensure all embedding files are available.")
        return
    
    # Confirm experiment parameters
    total_games = len(list(itertools.combinations(MODELS.keys(), 2))) * 20
    estimated_time = total_games * 0.5  # Rough estimate: 30 seconds per game
    
    print(f"Experiment Parameters:")
    print(f"  Models: {len(MODELS)} ({', '.join(MODELS.keys())})")
    print(f"  Combinations: {len(list(itertools.combinations(MODELS.keys(), 2)))}")
    print(f"  Games per combination: 20 (10 each direction)")
    print(f"  Total games: {total_games}")
    print(f"  Estimated time: {estimated_time/60:.0f} minutes")
    print()
    
    confirm = input("Start experiments? (y/n): ").lower().strip()
    if not confirm.startswith('y'):
        print("Experiments cancelled.")
        return
    
    # Run experiments
    experiments = BatchGameExperiments()
    experiments.run_all_experiments()


if __name__ == "__main__":
    main() 