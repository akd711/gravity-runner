import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import seaborn as sns
import pandas as pd
from datetime import datetime
import csv
import json
import ast
import os

# Set font support for English
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Set seaborn style
sns.set_style("whitegrid")
sns.set_palette("husl")
class SceneAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.raw_data = self.load_csv_data()
        self.scene_times = self.parse_scene_data()
        self.scene_durations = self.calculate_scene_durations()
    
    def load_csv_data(self):
        """Load data from CSV file"""
        try:
            # Increase field size limit to handle large data
            import sys
            maxInt = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(maxInt)
                    break
                except OverflowError:
                    maxInt = int(maxInt/10)
            
            all_custom_events = []
            all_ability_events = []
            player_count = 0
            
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse Custom_events column
                    custom_events_str = row.get('Custom_events', '')
                    if custom_events_str and custom_events_str.strip():
                        try:
                            # Try to parse as JSON array
                            custom_events = json.loads(custom_events_str)
                        except json.JSONDecodeError:
                            try:
                                # Try using ast.literal_eval
                                custom_events = ast.literal_eval(custom_events_str)
                            except (ValueError, SyntaxError):
                                # If all fails, skip this row
                                continue
                        
                        # Parse events_ability column
                        ability_events_str = row.get('events_ability', '')
                        if ability_events_str and ability_events_str.strip():
                            try:
                                # Try to parse as JSON array
                                ability_events = json.loads(ability_events_str)
                            except json.JSONDecodeError:
                                try:
                                    # Try using ast.literal_eval
                                    ability_events = ast.literal_eval(ability_events_str)
                                except (ValueError, SyntaxError):
                                    ability_events = []
                        else:
                            ability_events = []
                        
                        # Add player ID to each event
                        player_id = row.get('Player_ID', 'Unknown')
                        if player_id == 'Unknown':
                            # Try alternative column names
                            player_id = row.get('player_id', row.get('PlayerID', row.get('playerID', 'Unknown')))
                        
                        player_count += 1
                        
                        # Debug: Print first few rows to see column names
                        if player_count <= 3:
                            print(f"Row {player_count}: Player_ID='{player_id}', columns={list(row.keys())}")
                        
                        for event in custom_events:
                            if isinstance(event, str):
                                all_custom_events.append(f"[Player {player_id}] {event}")
                            else:
                                all_custom_events.append(f"[Player {player_id}] {str(event)}")
                        
                        for event in ability_events:
                            if isinstance(event, str):
                                all_ability_events.append(f"[Player {player_id}] {event}")
                            else:
                                all_ability_events.append(f"[Player {player_id}] {str(event)}")
            
            print(f"Loaded {len(all_custom_events)} custom events from {player_count} players")
            print(f"Loaded {len(all_ability_events)} ability events from {player_count} players")
            
            # Store ability events separately
            self.ability_events_data = all_ability_events
            
            return all_custom_events
            
        except Exception as e:
            print(f"Error: Cannot read file {self.csv_file}: {e}")
            return []
        
    def parse_scene_data(self):
        """Parse scene data, extract scene names and times"""
        scene_times = []
        player_scenes = defaultdict(list)  # Track scenes per player
        
        scene_events_found = 0
        
        for entry in self.raw_data:
            # Extract player ID and event
            player_match = re.search(r'\[Player (\w+)\] (.+)', entry)
            if not player_match:
                continue
                
            player_id = player_match.group(1)
            event = player_match.group(2)
            
            # Look for scene entry events
            match = re.search(r'Entered scene: (\w+) at ([\d.]+)s', event)
            if match:
                scene_name = match.group(1)
                time = float(match.group(2))
                scene_times.append((scene_name, time))
                player_scenes[player_id].append((scene_name, time))
                scene_events_found += 1
        
        # Sort by time to ensure chronological order
        scene_times.sort(key=lambda x: x[1])
        
        # Store player-specific data
        self.player_scenes = player_scenes
        
        print(f"Found {scene_events_found} scene entry events")
        print(f"Players with scene data: {len(player_scenes)}")
        for player_id, scenes in player_scenes.items():
            print(f"  Player {player_id}: {len(scenes)} scene visits")
        
        return scene_times
    
    def calculate_scene_durations(self):
        """Calculate duration for each scene"""
        scene_durations = defaultdict(list)
        
        for i in range(len(self.scene_times) - 1):
            current_scene, current_time = self.scene_times[i]
            next_scene, next_time = self.scene_times[i + 1]
            
            duration = next_time - current_time
            scene_durations[current_scene].append(duration)
        
        return scene_durations
    
    def analyze_player_exits(self):
        """Analyze where players exit the game (last scene visited)"""
        player_exits = {}
        player_scene_counts = {}
        
        for player_id, scenes in self.player_scenes.items():
            if scenes:
                # Sort scenes by time for this player
                scenes.sort(key=lambda x: x[1])
                
                # Count unique scenes visited
                unique_scenes = set(scene for scene, _ in scenes)
                player_scene_counts[player_id] = len(unique_scenes)
                
                # Last scene visited is the exit point
                last_scene, last_time = scenes[-1]
                player_exits[player_id] = {
                    'exit_scene': last_scene,
                    'exit_time': last_time,
                    'total_scenes_visited': len(unique_scenes),
                    'total_visits': len(scenes)
                }
        
        return player_exits, player_scene_counts
    
    def analyze_player_retention(self):
        """Analyze player retention based on exit scenes"""
        # Define the expected game progression order
        game_progression = [
            'Tutorial1', 'Tutorial2', 'Tutorial3', 
            'DashPuzzle', 'DashCombat', 
            'ShieldPuzzle', 'ShieldCombat', 
            'GravityFieldPuzzle', 'GravityFieldCombat', 
            'BossFight'
        ]
        
        # Create a mapping from scene to progress index
        scene_to_progress = {scene: idx for idx, scene in enumerate(game_progression)}
        
        # Count exits by scene
        exit_counts = defaultdict(int)
        for player_data in self.analyze_player_exits()[0].values():
            exit_scene = player_data['exit_scene']
            exit_counts[exit_scene] += 1
        
        # Calculate retention rates based on exits
        total_players = len(self.player_scenes)
        retention_rates = {}
        
        # Start with 100% retention at the beginning
        current_retention = 100.0
        
        for i, scene in enumerate(game_progression):
            # Calculate how many players exited at this scene
            exits_at_scene = exit_counts.get(scene, 0)
            
            # Retention rate is the percentage of players who continued past this scene
            retention_rates[scene] = current_retention
            
            # Update retention for next scene (subtract exits at current scene)
            if total_players > 0:
                current_retention -= (exits_at_scene / total_players) * 100
        
        return retention_rates, exit_counts, game_progression
    
    def analyze_ability_events(self):
        """Analyze ability events by scene"""
        ability_events = defaultdict(lambda: defaultdict(int))
        ability_events_found = 0
        
        # Use the ability events data instead of raw_data
        for entry in self.ability_events_data:
            # Extract player ID and event
            player_match = re.search(r'\[Player (\w+)\] (.+)', entry)
            if not player_match:
                continue
                
            player_id = player_match.group(1)
            event = player_match.group(2)
            
            # Look for ability events - single event format
            # Pattern: used_ability_name at (x, y, z) after time in scene 'scene_name'
            match = re.search(r'used_(\w+) at \(([^)]+)\) after ([\d.]+)s in scene \'(\w+)\'', event)
            if match:
                ability_name = match.group(1)
                scene_name = match.group(4)
                ability_events[scene_name][ability_name] += 1
                ability_events_found += 1
                
                # Debug: Print first few ability events
                if ability_events_found <= 5:
                    print(f"Found ability event: {ability_name} in {scene_name}")
        
        print(f"Total ability events found: {ability_events_found}")
        
        # Debug: Print sample events to see the actual format
        print("\nSample events from ability_events_data:")
        for i, entry in enumerate(self.ability_events_data[:10]):
            if 'used_' in entry:
                print(f"  {i}: {entry}")
        
        # Also check for any events containing 'used'
        print("\nAll ability events containing 'used':")
        used_events = [entry for entry in self.ability_events_data if 'used_' in entry]
        for i, entry in enumerate(used_events[:10]):
            print(f"  {i}: {entry}")
        
        return ability_events
    
    def get_statistics(self):
        """Get statistics information"""
        stats = {}
        
        for scene, durations in self.scene_durations.items():
            total_time = sum(durations)
            visit_count = len(durations)
            avg_time = total_time / visit_count if visit_count > 0 else 0
            min_time = min(durations) if durations else 0
            max_time = max(durations) if durations else 0
            
            stats[scene] = {
                'total_time': total_time,
                'visit_count': visit_count,
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'durations': durations
            }
        
        return stats
    
    def create_comprehensive_visualizations(self):
        """Create comprehensive visualization charts"""
        stats = self.get_statistics()
        
        # Create 2x3 subplot layout
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Pie Chart - Time Distribution
        ax1 = plt.subplot(2, 3, 1)
        self.create_pie_chart(ax1, stats)
        
        # 2. Bar Chart - Total Time
        ax2 = plt.subplot(2, 3, 2)
        self.create_total_time_bar_chart(ax2, stats)
        
        # 3. Bar Chart - Visit Count
        ax3 = plt.subplot(2, 3, 3)
        self.create_visit_count_bar_chart(ax3, stats)
        
        # 4. Exit Rate by Scene
        ax4 = plt.subplot(2, 3, 4)
        self.create_exit_rate_chart(ax4)
        
        # 5. Ability Events by Scene
        ax5 = plt.subplot(2, 3, 5)
        self.create_ability_events_chart(ax5)
        
        # 6. Player Retention Rate
        ax6 = plt.subplot(2, 3, 6)
        self.create_retention_chart(ax6)
        
        plt.tight_layout()
        return fig
    
    def create_pie_chart(self, ax, stats):
        """Create pie chart"""
        valid_data = [(scene, data['total_time']) for scene, data in stats.items() if data['total_time'] > 0]
        
        if valid_data:
            scenes, times = zip(*valid_data)
            colors = plt.cm.Set3(np.linspace(0, 1, len(scenes)))
            
            wedges, texts, autotexts = ax.pie(times, labels=scenes, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            
            for text in texts:
                text.set_color('black')
                text.set_fontsize(9)
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            ax.set_title('Scene Time Distribution', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scene Time Distribution', fontsize=12, fontweight='bold')
    
    def create_total_time_bar_chart(self, ax, stats):
        """Create total time bar chart"""
        scenes = list(stats.keys())
        times = [stats[scene]['total_time'] for scene in scenes]
        
        # Sort by time in ascending order
        sorted_data = sorted(zip(scenes, times), key=lambda x: x[1])
        scenes, times = zip(*sorted_data)
        
        bars = ax.bar(scenes, times, color='skyblue', edgecolor='navy', alpha=0.7)
        
        for bar, time in zip(bars, times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.01,
                   f'{time:.1f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        ax.set_title('Total Time by Scene (Ascending)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Time (seconds)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def create_visit_count_bar_chart(self, ax, stats):
        """Create visit count bar chart"""
        scenes = list(stats.keys())
        counts = [stats[scene]['visit_count'] for scene in scenes]
        
        # Sort by count in descending order
        sorted_data = sorted(zip(scenes, counts), key=lambda x: x[1], reverse=True)
        scenes, counts = zip(*sorted_data)
        
        bars = ax.bar(scenes, counts, color='lightcoral', edgecolor='darkred', alpha=0.7)
        
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Visit Count by Scene (Descending)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Visit Count', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def create_heatmap(self, ax, stats):
        """Create heatmap"""
        # Prepare data
        scenes = list(stats.keys())
        max_visits = max([stats[scene]['visit_count'] for scene in scenes])
        
        # Create heatmap data
        heatmap_data = []
        for scene in scenes:
            durations = stats[scene]['durations']
            # Fill to max visit count
            while len(durations) < max_visits:
                durations.append(0)
            heatmap_data.append(durations[:max_visits])
        
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(max_visits))
        ax.set_xticklabels([f'Visit {i+1}' for i in range(max_visits)])
        ax.set_yticks(range(len(scenes)))
        ax.set_yticklabels(scenes)
        
        # Add value labels
        for i in range(len(scenes)):
            for j in range(max_visits):
                if j < len(stats[scenes[i]]['durations']):
                    time = stats[scenes[i]]['durations'][j]
                    ax.text(j, i, f'{time:.1f}', ha='center', va='center', 
                           color='white' if time > 10 else 'black', fontweight='bold')
        
        ax.set_title('Scene Visit Time Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, label='Time (seconds)')
    
    def create_boxplot(self, ax, stats):
        """Create box plot"""
        data = []
        labels = []
        
        for scene, scene_stats in stats.items():
            if scene_stats['durations']:
                data.append(scene_stats['durations'])
                labels.append(scene)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Set colors
            colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title('Scene Time Distribution Box Plot', fontsize=12, fontweight='bold')
            ax.set_ylabel('Time (seconds)', fontsize=10)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Scene Time Distribution Box Plot', fontsize=12, fontweight='bold')
    
    def create_timeline(self, ax):
        """Create timeline chart"""
        if len(self.scene_times) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Game Timeline', fontsize=12, fontweight='bold')
            return
        
        times = [time for _, time in self.scene_times]
        scenes = [scene for scene, _ in self.scene_times]
        
        # Create timeline
        ax.plot(times, range(len(times)), 'o-', linewidth=2, markersize=8)
        
        # Add scene labels
        for i, (scene, time) in enumerate(zip(scenes, times)):
            ax.annotate(scene, (time, i), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax.set_title('Game Timeline', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (seconds)', fontsize=10)
        ax.set_ylabel('Scene Order', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def create_exit_distribution_chart(self, ax):
        """Create player exit distribution chart"""
        player_exits, _ = self.analyze_player_exits()
        
        if not player_exits:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Player Exit Distribution', fontsize=12, fontweight='bold')
            return
        
        # Count exits by scene
        exit_scenes = defaultdict(int)
        for player_data in player_exits.values():
            exit_scenes[player_data['exit_scene']] += 1
        
        if exit_scenes:
            scenes = list(exit_scenes.keys())
            counts = list(exit_scenes.values())
            
            # Sort by count in descending order
            sorted_data = sorted(zip(scenes, counts), key=lambda x: x[1], reverse=True)
            scenes, counts = zip(*sorted_data)
            
            bars = ax.bar(scenes, counts, color='red', alpha=0.7)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Player Exit Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Exit Scene')
            ax.set_ylabel('Number of Players')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Player Exit Distribution', fontsize=12, fontweight='bold')
    
    def create_exit_rate_chart(self, ax):
        """Create exit rate by scene chart"""
        player_exits, _ = self.analyze_player_exits()
        
        if not player_exits:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Exit Rate by Scene', fontsize=12, fontweight='bold')
            return
        
        # Count exits by scene
        exit_scenes = defaultdict(int)
        for player_data in player_exits.values():
            exit_scenes[player_data['exit_scene']] += 1
        
        if exit_scenes:
            total_players = len(self.player_scenes)
            scenes = list(exit_scenes.keys())
            rates = [(exit_scenes[scene] / total_players) * 100 for scene in scenes]
            
            # Sort by exit rate in descending order
            sorted_data = sorted(zip(scenes, rates), key=lambda x: x[1], reverse=True)
            scenes, rates = zip(*sorted_data)
            
            bars = ax.bar(scenes, rates, color='red', alpha=0.7)
            
            # Add value labels
            for bar, rate in zip(bars, rates):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Exit Rate by Scene (Descending)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Scene')
            ax.set_ylabel('Exit Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Exit Rate by Scene', fontsize=12, fontweight='bold')
    
    def create_exit_count_chart(self, ax):
        """Create exit count by scene chart"""
        player_exits, _ = self.analyze_player_exits()
        
        if not player_exits:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Exit Count by Scene', fontsize=12, fontweight='bold')
            return
        
        # Count exits by scene
        exit_scenes = defaultdict(int)
        for player_data in player_exits.values():
            exit_scenes[player_data['exit_scene']] += 1
        
        if exit_scenes:
            scenes = list(exit_scenes.keys())
            counts = list(exit_scenes.values())
            
            # Sort by count in descending order
            sorted_data = sorted(zip(scenes, counts), key=lambda x: x[1], reverse=True)
            scenes, counts = zip(*sorted_data)
            
            bars = ax.bar(scenes, counts, color='darkred', alpha=0.7)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       str(count), ha='center', va='bottom', fontweight='bold')
            
            ax.set_title('Exit Count by Scene (Descending)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Scene')
            ax.set_ylabel('Exit Count')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No exit data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Exit Count by Scene', fontsize=12, fontweight='bold')
    
    def create_ability_events_chart(self, ax):
        """Create ability events by scene chart"""
        ability_events = self.analyze_ability_events()
        
        if not ability_events:
            ax.text(0.5, 0.5, 'No ability events found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ability Events by Scene', fontsize=12, fontweight='bold')
            return
        
        # Define game progression order (same as retention chart)
        game_progression = [
            'Tutorial1', 'Tutorial2', 'Tutorial3', 
            'DashPuzzle', 'DashCombat', 
            'ShieldPuzzle', 'ShieldCombat', 
            'GravityFieldPuzzle', 'GravityFieldCombat', 
            'BossFight'
        ]
        
        # Filter scenes that have ability events and sort by game progression
        scenes_with_events = [scene for scene in game_progression if scene in ability_events]
        
        if not scenes_with_events:
            ax.text(0.5, 0.5, 'No ability events found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ability Events by Scene', fontsize=12, fontweight='bold')
            return
        
        all_abilities = set()
        for scene_abilities in ability_events.values():
            all_abilities.update(scene_abilities.keys())
        
        # Define the three main abilities
        main_abilities = ['dash', 'shield', 'gravity_field']
        abilities_to_show = [ability for ability in main_abilities if ability in all_abilities]
        
        if not abilities_to_show:
            ax.text(0.5, 0.5, 'No main ability events found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Ability Events by Scene', fontsize=12, fontweight='bold')
            return
        
        # Prepare data for grouped bar chart
        x = np.arange(len(scenes_with_events))
        width = 0.25  # Width of each bar
        
        # Create grouped bars
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
        bars = []
        
        for i, ability in enumerate(abilities_to_show):
            counts = [ability_events[scene].get(ability, 0) for scene in scenes_with_events]
            # Create proper label for gravity_field
            if ability == 'gravity_field':
                label = 'Gravity Field'
            else:
                label = ability.capitalize()
            
            bar = ax.bar(x + i * width, counts, width, label=label, 
                        color=colors[i], alpha=0.8)
            bars.append(bar)
            
            # Add value labels on bars
            for j, count in enumerate(counts):
                if count > 0:
                    ax.text(x[j] + i * width, count + 0.1, str(count), 
                           ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # Customize the chart
        ax.set_title('Ability Events by Scene', fontsize=12, fontweight='bold')
        ax.set_xlabel('Scene', fontsize=10)
        ax.set_ylabel('Event Count', fontsize=10)
        ax.set_xticks(x + width)
        ax.set_xticklabels(scenes_with_events, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Adjust layout to prevent label cutoff
        plt.setp(ax.get_xticklabels(), ha='right', rotation=45)
    
    def create_retention_chart(self, ax):
        """Create player retention rate chart"""
        retention_rates, exit_counts, game_progression = self.analyze_player_retention()
        
        if not retention_rates:
            ax.text(0.5, 0.5, 'No retention data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Player Retention Rate', fontsize=12, fontweight='bold')
            return
        
        # Create the line chart
        scenes = list(retention_rates.keys())
        rates = list(retention_rates.values())
        
        ax.plot(range(len(scenes)), rates, 'o-', linewidth=2, markersize=8, color='blue')
        
        # Add value labels
        for i, rate in enumerate(rates):
            ax.text(i, rate + 1, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # Set labels and title
        ax.set_title('Player Retention Rate by Progress', fontsize=12, fontweight='bold')
        ax.set_xlabel('Game Progress', fontsize=10)
        ax.set_ylabel('Retention Rate (%)', fontsize=10)
        ax.set_xticks(range(len(scenes)))
        ax.set_xticklabels(scenes, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0, 105)
    
    def print_detailed_report(self):
        """Print detailed report"""
        stats = self.get_statistics()
        player_exits, player_scene_counts = self.analyze_player_exits()
        retention_rates, exit_counts, game_progression = self.analyze_player_retention()
        ability_events = self.analyze_ability_events()
        
        print("=" * 80)
        print("🎮 Advanced Scene Time Analysis Report")
        print("=" * 80)
        print(f"📅 Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Total Scenes: {len(stats)}")
        print(f"📋 Total Records: {len(self.scene_times)}")
        print(f"👥 Total Players: {len(self.player_scenes)}")
        
        total_time = sum([stats[scene]['total_time'] for scene in stats])
        total_visits = sum([stats[scene]['visit_count'] for scene in stats])
        
        print(f"⏱️ Total Game Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(f"🔄 Total Visits: {total_visits}")
        
        # Player exit analysis
        print("\n" + "=" * 80)
        print("🚪 Player Exit Analysis")
        print("=" * 80)
        
        if player_exits:
            # Count exits by scene
            exit_scenes = defaultdict(int)
            for player_data in player_exits.values():
                exit_scenes[player_data['exit_scene']] += 1
            
            print(f"📊 Exit Scene Distribution:")
            for scene, count in sorted(exit_scenes.items(), key=lambda x: x[1], reverse=True):
                print(f"   {scene}: {count} players")
            
            # Scene visit statistics
            scene_visit_counts = list(player_scene_counts.values())
            avg_scenes_visited = sum(scene_visit_counts) / len(scene_visit_counts) if scene_visit_counts else 0
            max_scenes_visited = max(scene_visit_counts) if scene_visit_counts else 0
            min_scenes_visited = min(scene_visit_counts) if scene_visit_counts else 0
            
            print(f"\n📈 Scene Visit Statistics:")
            print(f"   Average scenes visited per player: {avg_scenes_visited:.1f}")
            print(f"   Maximum scenes visited: {max_scenes_visited}")
            print(f"   Minimum scenes visited: {min_scenes_visited}")
            
            # Most common exit points
            most_common_exit = max(exit_scenes.items(), key=lambda x: x[1]) if exit_scenes else None
            if most_common_exit:
                print(f"   Most common exit scene: {most_common_exit[0]} ({most_common_exit[1]} players)")
        else:
            print("No player exit data found")
        
        # Player retention analysis
        print("\n" + "=" * 80)
        print("📈 Player Retention Analysis")
        print("=" * 80)
        
        if retention_rates:
            total_players = len(self.player_scenes)
            print(f"📊 Total Players: {total_players}")
            
            # Show retention at each progress level
            print(f"\n🎯 Retention by Progress Level:")
            for scene in game_progression:
                if scene in retention_rates:
                    retention_rate = retention_rates[scene]
                    exits_at_scene = exit_counts.get(scene, 0)
                    print(f"   {scene}: {retention_rate:.1f}% retention ({exits_at_scene} exits)")
            
            # Find the biggest drop-off point
            if len(exit_counts) > 1:
                biggest_drop = 0
                drop_scene = ""
                
                for scene, exits in exit_counts.items():
                    if exits > biggest_drop:
                        biggest_drop = exits
                        drop_scene = scene
                
                if biggest_drop > 0:
                    print(f"\n⚠️  Biggest drop-off: {drop_scene} (lost {biggest_drop} players)")
        else:
            print("No retention data found")
        
        # Ability events analysis
        print("\n" + "=" * 80)
        print("⚡ Ability Events Analysis")
        print("=" * 80)
        
        if ability_events:
            total_ability_events = 0
            for scene_abilities in ability_events.values():
                total_ability_events += sum(scene_abilities.values())
            
            print(f"📊 Total Ability Events: {total_ability_events}")
            print(f"🎯 Scenes with Ability Events: {len(ability_events)}")
            
            print(f"\n🎮 Ability Events by Scene:")
            for scene in sorted(ability_events.keys()):
                scene_abilities = ability_events[scene]
                total_in_scene = sum(scene_abilities.values())
                print(f"   {scene}: {total_in_scene} total events")
                for ability, count in sorted(scene_abilities.items()):
                    print(f"     - {ability}: {count} times")
        else:
            print("No ability events found")
        
        print("\n" + "=" * 80)
        print("📈 Detailed Scene Statistics")
        print("=" * 80)
        
        # Sort by total time
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for scene, data in sorted_stats:
            print(f"\n🎯 Scene: {scene}")
            print(f"   📊 Visit Count: {data['visit_count']}")
            print(f"   ⏱️ Total Time: {data['total_time']:.2f} seconds")
            print(f"   📈 Average Time: {data['avg_time']:.2f} seconds")
            print(f"   📉 Min Time: {data['min_time']:.2f} seconds")
            print(f"   📊 Max Time: {data['max_time']:.2f} seconds")
            
            if len(data['durations']) > 1:
                print(f"   📋 Detailed Records:")
                for i, duration in enumerate(data['durations'], 1):
                    print(f"      Visit {i}: {duration:.2f} seconds")
        
        # Find most visited and longest stay scenes
        most_visited = max(stats.items(), key=lambda x: x[1]['visit_count'])
        longest_stay = max(stats.items(), key=lambda x: x[1]['total_time'])
        shortest_stay = min(stats.items(), key=lambda x: x[1]['total_time'])
        
        print("\n" + "=" * 80)
        print("🏆 Key Metrics")
        print("=" * 80)
        print(f"🎯 Most Visited Scene: {most_visited[0]} ({most_visited[1]['visit_count']} times)")
        print(f"⏱️ Longest Stay Scene: {longest_stay[0]} ({longest_stay[1]['total_time']:.2f} seconds)")
        print(f"⚡ Shortest Stay Scene: {shortest_stay[0]} ({shortest_stay[1]['total_time']:.2f} seconds)")
        
        # Calculate efficiency metrics
        if total_visits > 0:
            avg_time_per_visit = total_time / total_visits
            print(f"📊 Average Time per Visit: {avg_time_per_visit:.2f} seconds")
        
        # Calculate scene transition frequency
        if len(self.scene_times) > 1:
            total_transitions = len(self.scene_times) - 1
            transition_frequency = total_transitions / (total_time / 60)  # transitions per minute
            print(f"🔄 Scene Transition Frequency: {transition_frequency:.2f} times/minute")

def main():
    """Main function"""
    csv_file = "data/UnityDataCollections.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: File {csv_file} does not exist")
        print("Please ensure the CSV file is in the data directory")
        return
    
    print("🔍 Starting advanced scene time analysis...")
    
    # Create analyzer
    analyzer = SceneAnalyzer(csv_file)
    
    # Print detailed report
    analyzer.print_detailed_report()
    
    # Create comprehensive visualization charts
    fig = analyzer.create_comprehensive_visualizations()
    
    # Create AnalysisResult directory if it doesn't exist
    analysis_dir = "AnalysisResult"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{analysis_dir}/advanced_scene_analysis_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n💾 Advanced analysis chart saved as '{filename}'")
    
    # Show chart
    plt.show()
    
    return analyzer

if __name__ == "__main__":
    analyzer = main() 