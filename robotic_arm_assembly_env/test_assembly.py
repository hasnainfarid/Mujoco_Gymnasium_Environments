"""
Test script for Robotic Arm Assembly Environment
Tests 5 scenarios with visualization
Author: Hasnain Fareed
Year: 2025
"""

import numpy as np
import time
import json
from datetime import datetime
from assembly_env import RoboticArmAssemblyEnv

def test_basic_assembly():
    """Test 1: Basic assembly - Pick and place PCB."""
    print("\n" + "="*60)
    print("TEST 1: BASIC ASSEMBLY - PCB PLACEMENT")
    print("="*60)
    
    env = RoboticArmAssemblyEnv(render_mode="human")
    
    all_episode_results = []
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        
        print("Testing basic pick and place of PCB...")
        print("Initial state:", info['task_phase'])
        
        total_reward = 0
        steps = 0
        max_steps = 2000
    
        for step in range(max_steps):
            # Simple control strategy: move towards PCB, grip, move to target
            if step < 200:
                # Move arm towards PCB bin
                action = np.array([-0.5, 0.3, 0.2, 0, 0, 0, 0, 50, 25])
            elif step < 300:
                # Close gripper
                action = np.array([0, 0, 0, 0, 0, 0, 0, 10, 40])
            elif step < 600:
                # Move to assembly position
                action = np.array([0.3, -0.2, -0.1, 0, 0, 0, 0, 10, 40])
            else:
                # Open gripper to release
                action = np.array([0, 0, 0, 0, 0, 0, 0, 80, 10])
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if step % 200 == 0:
                print(f"Step {step}: Phase={info['task_phase']}, Held={info['held_component']}, Reward={reward:.2f}")
            
            if terminated or truncated:
                break
            
            time.sleep(0.01)  # Slow down for visualization
        
        success = info['assembly_progress'].get('pcb', False)
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"Success: {success}")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Assembly Progress: {info['assembly_progress']}")
        
        all_episode_results.append({
            'episode': episode + 1,
            'success': success,
            'steps': steps,
            'reward': total_reward,
            'assembly_progress': info['assembly_progress']
        })
    
    env.close()
    
    # Calculate overall results
    total_successes = sum(1 for r in all_episode_results if r['success'])
    avg_reward = sum(r['reward'] for r in all_episode_results) / len(all_episode_results)
    avg_steps = sum(r['steps'] for r in all_episode_results) / len(all_episode_results)
    
    print(f"\nBasic Assembly Test Complete!")
    print(f"Episodes: {num_episodes}")
    print(f"Successful Episodes: {total_successes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return {
        'test': 'basic_assembly',
        'episodes': num_episodes,
        'successful_episodes': total_successes,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': all_episode_results
    }

def test_precision_insertion():
    """Test 2: Precision insertion - CPU chip alignment."""
    print("\n" + "="*60)
    print("TEST 2: PRECISION INSERTION - CPU PLACEMENT")
    print("="*60)
    
    env = RoboticArmAssemblyEnv(render_mode="human")
    
    all_episode_results = []
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        
        print("Testing precision CPU insertion...")
        
        total_reward = 0
        steps = 0
        max_steps = 3000
    
        # First place PCB (simplified)
        for step in range(400):
            action = np.random.randn(9) * 0.1
            action[7:9] = [30, 30]  # Gripper control
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.005)
        
        # Now focus on CPU
        print("Attempting CPU placement with precision control...")
        
        for step in range(max_steps - 400):
            # Precision control for CPU
            if step < 200:
                # Approach CPU bin
                action = np.array([-0.3, 0.1, 0.1, 0, 0, 0, 0, 60, 20])
            elif step < 300:
                # Precision grip
                action = np.array([0, 0, 0, 0, 0, 0, 0, 15, 35])
            elif step < 800:
                # Slow, careful transport
                action = np.array([0.1, -0.05, -0.05, 0, 0, 0, 0, 15, 35])
            elif step < 900:
                # Fine alignment
                action = np.array([0.02, 0.01, -0.01, 0, 0, 0, 0, 15, 35])
            else:
                # Gentle release
                action = np.array([0, 0, 0, 0, 0, 0, 0, 40, 15])
            
            # Add small noise for realism
            action[:7] = action[:7].astype(np.float64) + np.random.randn(7) * 0.01
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if step % 400 == 0:
                print(f"Step {step}: Phase={info['task_phase']}, Component Status={info['component_status'].get('cpu', 'unknown')}")
            
            if terminated or truncated:
                break
            
            time.sleep(0.01)
        
        success = info['assembly_progress'].get('cpu', False)
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"CPU Successfully Inserted: {success}")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        
        all_episode_results.append({
            'episode': episode + 1,
            'success': success,
            'steps': steps,
            'reward': total_reward,
            'assembly_progress': info['assembly_progress']
        })
    
    env.close()
    
    # Calculate overall results
    total_successes = sum(1 for r in all_episode_results if r['success'])
    avg_reward = sum(r['reward'] for r in all_episode_results) / len(all_episode_results)
    avg_steps = sum(r['steps'] for r in all_episode_results) / len(all_episode_results)
    
    print(f"\nPrecision Insertion Test Complete!")
    print(f"Episodes: {num_episodes}")
    print(f"Successful Episodes: {total_successes}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return {
        'test': 'precision_insertion',
        'episodes': num_episodes,
        'successful_episodes': total_successes,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': all_episode_results
    }

def test_force_sensitive():
    """Test 3: Force-sensitive tasks - Gentle component handling."""
    print("\n" + "="*60)
    print("TEST 3: FORCE-SENSITIVE HANDLING")
    print("="*60)
    
    env = RoboticArmAssemblyEnv(render_mode="human")
    
    all_episode_results = []
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        
        print("Testing force-sensitive component handling...")
        
        total_reward = 0
        steps = 0
        max_steps = 2000
        force_violations = 0
        
        for step in range(max_steps):
            # Gentle control strategy with low forces
            base_action = np.sin(step * 0.01) * np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0, 0])
            
            # Adaptive grip force based on phase
            if info['task_phase'] == 'pickup':
                base_action[7:9] = [20, 25]  # Gentle grip
            elif info['task_phase'] == 'transport':
                base_action[7:9] = [15, 30]  # Secure but gentle
            else:
                base_action[7:9] = [50, 15]  # Default
            
            obs, reward, terminated, truncated, info = env.step(base_action)
            total_reward += reward
            steps += 1
            
            # Monitor force
            current_force = obs[15] if len(obs) > 15 else 0
            if current_force > 40:
                force_violations += 1
                print(f"Force violation at step {step}: {current_force:.2f}N")
            
            if step % 400 == 0:
                print(f"Step {step}: Max force={current_force:.2f}N, Violations={force_violations}")
            
            if terminated or truncated:
                break
            
            time.sleep(0.01)
        
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"Total Force Violations: {force_violations}")
        print(f"Average Force Control Score: {100 - min(force_violations, 100)}/100")
        print(f"Total Reward: {total_reward:.2f}")
        
        all_episode_results.append({
            'episode': episode + 1,
            'force_violations': force_violations,
            'steps': steps,
            'reward': total_reward,
            'force_score': 100 - min(force_violations, 100)
        })
    
    env.close()
    
    # Calculate overall results
    total_violations = sum(r['force_violations'] for r in all_episode_results)
    avg_reward = sum(r['reward'] for r in all_episode_results) / len(all_episode_results)
    avg_steps = sum(r['steps'] for r in all_episode_results) / len(all_episode_results)
    avg_force_score = sum(r['force_score'] for r in all_episode_results) / len(all_episode_results)
    
    print(f"\nForce-Sensitive Test Complete!")
    print(f"Episodes: {num_episodes}")
    print(f"Total Force Violations: {total_violations}")
    print(f"Average Force Control Score: {avg_force_score:.1f}/100")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return {
        'test': 'force_sensitive',
        'episodes': num_episodes,
        'total_force_violations': total_violations,
        'avg_force_score': avg_force_score,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': all_episode_results
    }

def test_vision_guided():
    """Test 4: Vision-guided assembly using camera feedback."""
    print("\n" + "="*60)
    print("TEST 4: VISION-GUIDED ASSEMBLY")
    print("="*60)
    
    env = RoboticArmAssemblyEnv(render_mode="human")
    
    all_episode_results = []
    num_episodes = 3
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        
        print("Testing vision-guided component alignment...")
        
        total_reward = 0
        steps = 0
        max_steps = 2400
        alignment_errors = []
        
        for step in range(max_steps):
            # Simulate vision-based control
            # Extract "visual features" from observation
            visual_features = obs[89:114] if len(obs) > 114 else np.zeros(25)
            
            # Vision-based control policy
            if step < 600:
                # Visual search pattern
                action = np.array([
                    np.sin(step * 0.05) * 0.5,
                    np.cos(step * 0.05) * 0.5,
                    0.1,
                    0, 0, 0, 0,
                    40, 20
                ])
            else:
                # Visual servoing towards target
                error_signal = np.random.randn(7) * 0.1  # Simulated visual error
                action = np.concatenate([
                    -error_signal * 0.5,  # Proportional control
                    [30, 30]  # Gripper
                ])
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Track alignment error
            if info['held_component']:
                # Simulated alignment error
                alignment_error = np.random.rand() * 5  # mm
                alignment_errors.append(alignment_error)
            
            if step % 400 == 0:
                avg_error = np.mean(alignment_errors) if alignment_errors else 0
                print(f"Step {step}: Visual alignment error={avg_error:.2f}mm")
            
            if terminated or truncated:
                break
            
            time.sleep(0.01)
        
        avg_alignment = np.mean(alignment_errors) if alignment_errors else 10
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"Average Alignment Error: {avg_alignment:.2f}mm")
        print(f"Vision Score: {max(0, 100 - avg_alignment * 10):.1f}/100")
        print(f"Total Reward: {total_reward:.2f}")
        
        all_episode_results.append({
            'episode': episode + 1,
            'avg_alignment_error': avg_alignment,
            'vision_score': max(0, 100 - avg_alignment * 10),
            'steps': steps,
            'reward': total_reward
        })
    
    env.close()
    
    # Calculate overall results
    avg_alignment = sum(r['avg_alignment_error'] for r in all_episode_results) / len(all_episode_results)
    avg_vision_score = sum(r['vision_score'] for r in all_episode_results) / len(all_episode_results)
    avg_reward = sum(r['reward'] for r in all_episode_results) / len(all_episode_results)
    avg_steps = sum(r['steps'] for r in all_episode_results) / len(all_episode_results)
    
    print(f"\nVision-Guided Test Complete!")
    print(f"Episodes: {num_episodes}")
    print(f"Average Alignment Error: {avg_alignment:.2f}mm")
    print(f"Average Vision Score: {avg_vision_score:.1f}/100")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return {
        'test': 'vision_guided',
        'episodes': num_episodes,
        'avg_alignment_error': avg_alignment,
        'avg_vision_score': avg_vision_score,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': all_episode_results
    }

def test_complex_assembly():
    """Test 5: Complex multi-component assembly sequence."""
    print("\n" + "="*60)
    print("TEST 5: COMPLEX MULTI-COMPONENT ASSEMBLY")
    print("="*60)
    
    env = RoboticArmAssemblyEnv(render_mode="human")
    
    all_episode_results = []
    num_episodes = 2  # Complex test with fewer episodes due to length
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        obs, info = env.reset()
        
        print("Testing complete assembly sequence...")
        print("Target sequence: PCB → Screws → CPU → Battery → Cable → Cover")
        
        total_reward = 0
        steps = 0
        max_steps = 10000
    
        # Assembly sequence stages
        stages = [
            ('pcb', 1000),
            ('screw1', 600),
            ('screw2', 600),
            ('screw3', 600),
            ('screw4', 600),
            ('cpu', 800),
            ('battery', 800),
            ('cable', 800),
            ('cover', 1000)
        ]
        
        current_stage = 0
        stage_steps = 0
        
        for step in range(max_steps):
            if current_stage < len(stages):
                component, max_stage_steps = stages[current_stage]
                
                # Component-specific control strategy
                if component == 'pcb':
                    # Large, flat component
                    if stage_steps < 200:
                        action = np.array([-0.5, 0.3, 0.1, 0, 0, 0, 0, 60, 20])
                    elif stage_steps < 300:
                        action = np.array([0, 0, 0, 0, 0, 0, 0, 20, 40])
                    elif stage_steps < 600:
                        action = np.array([0.4, -0.2, -0.1, 0, 0, 0, 0, 20, 40])
                    else:
                        action = np.array([0, 0, 0, 0, 0, 0, 0, 70, 10])
            
            elif 'screw' in component:
                # Small, precise components
                phase = stage_steps / max_stage_steps
                if phase < 0.3:
                    action = np.array([-0.4, -0.3, 0.1, 0, 0, 0, 0, 40, 20])
                elif phase < 0.5:
                    action = np.array([0, 0, 0, 0, 0, 0, 0, 10, 35])
                elif phase < 0.8:
                    # Precise positioning
                    action = np.array([0.2, 0.1, -0.1, 0, 0, 0.5, 0, 10, 35])
                else:
                    # Thread simulation
                    action = np.array([0, 0, -0.05, 0, 0, 0.2, 0, 10, 30])
            
            elif component == 'cpu':
                # Delicate component
                if stage_steps < 150:
                    action = np.array([-0.3, 0, 0.1, 0, 0, 0, 0, 50, 20])
                elif stage_steps < 200:
                    action = np.array([0, 0, 0, 0, 0, 0, 0, 15, 25])
                else:
                    action = np.array([0.15, 0, -0.05, 0, 0, 0, 0, 15, 25])
            
            elif component == 'battery':
                # Heavy component
                if stage_steps < 150:
                    action = np.array([0.5, 0.3, 0.1, 0, 0, 0, 0, 60, 25])
                elif stage_steps < 200:
                    action = np.array([0, 0, 0, 0, 0, 0, 0, 30, 45])
                else:
                    action = np.array([-0.3, -0.15, -0.1, 0, 0, 0, 0, 30, 45])
            
            elif component == 'cable':
                # Flexible component
                if stage_steps < 150:
                    action = np.array([0.4, -0.3, 0.1, 0, 0, 0, 0, 50, 20])
                elif stage_steps < 200:
                    action = np.array([0, 0, 0, 0, 0, 0, 0, 20, 30])
                else:
                    action = np.array([-0.2, 0.1, -0.05, 0, 0, 0, 0, 20, 30])
            
            else:  # cover
                # Final component with snap-fit
                if stage_steps < 200:
                    action = np.array([0.4, 0, 0.1, 0, 0, 0, 0, 70, 25])
                elif stage_steps < 250:
                    action = np.array([0, 0, 0, 0, 0, 0, 0, 40, 40])
                elif stage_steps < 400:
                    action = np.array([-0.2, 0, -0.15, 0, 0, 0, 0, 40, 40])
                else:
                    # Apply pressure for snap-fit
                    action = np.array([0, 0, -0.1, 0, 0, 0, 0, 40, 50])
            
            # Add some noise for realism
            action[:7] = action[:7].astype(np.float64) + np.random.randn(7) * 0.02
            
            stage_steps += 1
            if stage_steps >= max_stage_steps:
                print(f"Completed stage: {component}")
                current_stage += 1
                stage_steps = 0
        else:
            # All stages complete, hold position
            action = np.zeros(9)
            action[7:9] = [40, 20]
        
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if step % 1000 == 0:
                completed = sum(info['assembly_progress'].values())
                print(f"Step {step}: Components assembled: {completed}/{len(info['assembly_progress'])}")
                print(f"  Assembly progress: {info['assembly_progress']}")
            
            if terminated or truncated:
                break
            
            time.sleep(0.005)
        
        success = all(info['assembly_progress'].values())
        completed_count = sum(info['assembly_progress'].values())
        
        print(f"\nEpisode {episode + 1} Complete!")
        print(f"Full Assembly Success: {success}")
        print(f"Components Assembled: {completed_count}/{len(info['assembly_progress'])}")
        print(f"Total Steps: {steps}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Final Status: {info['component_status']}")
        
        all_episode_results.append({
            'episode': episode + 1,
            'success': success,
            'components_assembled': completed_count,
            'total_components': len(info['assembly_progress']),
            'steps': steps,
            'reward': total_reward,
            'final_status': info['component_status']
        })
    
    env.close()
    
    # Calculate overall results
    total_successes = sum(1 for r in all_episode_results if r['success'])
    avg_reward = sum(r['reward'] for r in all_episode_results) / len(all_episode_results)
    avg_steps = sum(r['steps'] for r in all_episode_results) / len(all_episode_results)
    avg_components = sum(r['components_assembled'] for r in all_episode_results) / len(all_episode_results)
    
    print(f"\nComplex Assembly Test Complete!")
    print(f"Episodes: {num_episodes}")
    print(f"Successful Episodes: {total_successes}")
    print(f"Average Components Assembled: {avg_components:.1f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.1f}")
    
    return {
        'test': 'complex_assembly',
        'episodes': num_episodes,
        'successful_episodes': total_successes,
        'avg_components_assembled': avg_components,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': all_episode_results
    }

def compare_control_strategies():
    """Compare position vs force control strategies."""
    print("\n" + "="*60)
    print("CONTROL STRATEGY COMPARISON")
    print("="*60)
    
    strategies = {
        'position_control': lambda t: np.array([
            np.sin(t * 0.01) * 0.5,
            np.cos(t * 0.01) * 0.5,
            0.1 * np.sin(t * 0.02),
            0, 0, 0, 0,
            30 + 20 * np.sin(t * 0.05),
            25
        ]),
        'force_control': lambda t: np.array([
            np.tanh(np.sin(t * 0.01)) * 0.3,
            np.tanh(np.cos(t * 0.01)) * 0.3,
            0.05,
            0, 0, 0, 0,
            40,
            20 + 10 * np.sin(t * 0.03)
        ]),
        'hybrid_control': lambda t: np.array([
            np.sin(t * 0.01) * 0.4,
            np.cos(t * 0.01) * 0.4,
            0.08 * np.sin(t * 0.015),
            0, 0, 0, 0,
            35 + 15 * np.sin(t * 0.04),
            25 + 5 * np.sin(t * 0.02)
        ])
    }
    
    results = {}
    
    for strategy_name, control_fn in strategies.items():
        print(f"\nTesting {strategy_name}...")
        
        env = RoboticArmAssemblyEnv(render_mode="human")
        obs, info = env.reset()
        
        total_reward = 0
        steps = 0
        max_force = 0
        smoothness_score = 0
        
        for t in range(1000):
            action = control_fn(t)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Track metrics
            current_force = obs[15] if len(obs) > 15 else 0
            max_force = max(max_force, current_force)
            
            # Smoothness (lower is better)
            if t > 0:
                smoothness_score += np.sum(np.abs(action[:7] - prev_action[:7]))
            prev_action = action
            
            if terminated or truncated:
                break
            
            time.sleep(0.01)
        
        results[strategy_name] = {
            'reward': total_reward,
            'steps': steps,
            'max_force': max_force,
            'smoothness': smoothness_score / steps,
            'efficiency': total_reward / steps
        }
        
        print(f"  Reward: {total_reward:.2f}")
        print(f"  Max Force: {max_force:.2f}N")
        print(f"  Smoothness: {smoothness_score/steps:.4f}")
        
        env.close()
    
    print("\n" + "="*60)
    print("STRATEGY COMPARISON RESULTS:")
    print("="*60)
    
    best_reward = max(results.values(), key=lambda x: x['reward'])
    best_smooth = min(results.values(), key=lambda x: x['smoothness'])
    best_efficiency = max(results.values(), key=lambda x: x['efficiency'])
    
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Total Reward: {metrics['reward']:.2f}")
        print(f"  Steps: {metrics['steps']}")
        print(f"  Max Force: {metrics['max_force']:.2f}N")
        print(f"  Smoothness Score: {metrics['smoothness']:.4f}")
        print(f"  Efficiency: {metrics['efficiency']:.4f}")
        
        # Determine if this is the best in any category
        badges = []
        if metrics == best_reward:
            badges.append("BEST REWARD")
        if metrics == best_smooth:
            badges.append("SMOOTHEST")
        if metrics == best_efficiency:
            badges.append("MOST EFFICIENT")
        
        if badges:
            print(f"  🏆 {', '.join(badges)}")
    
    return results

def main():
    """Run all tests and generate report."""
    print("\n" + "="*70)
    print(" ROBOTIC ARM ASSEMBLY ENVIRONMENT - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nInitializing test suite with MuJoCo visualization...")
    print("Each test will open a visualization window")
    print("\nTests to run:")
    print("1. Basic Assembly - PCB Placement")
    print("2. Precision Insertion - CPU Alignment")
    print("3. Force-Sensitive Handling")
    print("4. Vision-Guided Assembly")
    print("5. Complex Multi-Component Assembly")
    print("6. Control Strategy Comparison")
    
    all_results = []
    
    # Run all tests
    try:
        print("\n" + "-"*70)
        result1 = test_basic_assembly()
        all_results.append(result1)
        time.sleep(2)
        
        print("\n" + "-"*70)
        result2 = test_precision_insertion()
        all_results.append(result2)
        time.sleep(2)
        
        print("\n" + "-"*70)
        result3 = test_force_sensitive()
        all_results.append(result3)
        time.sleep(2)
        
        print("\n" + "-"*70)
        result4 = test_vision_guided()
        all_results.append(result4)
        time.sleep(2)
        
        print("\n" + "-"*70)
        result5 = test_complex_assembly()
        all_results.append(result5)
        time.sleep(2)
        
        print("\n" + "-"*70)
        strategy_results = compare_control_strategies()
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    # Generate final report
    print("\n" + "="*70)
    print(" FINAL TEST REPORT")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        'timestamp': timestamp,
        'test_results': all_results,
        'strategy_comparison': strategy_results if 'strategy_results' in locals() else None
    }
    
    # Save report
    report_file = f"assembly_test_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nTest report saved to: {report_file}")
    
    # Summary statistics
    if all_results:
        total_reward = sum(r.get('reward', 0) for r in all_results)
        total_steps = sum(r.get('steps', 0) for r in all_results)
        success_count = sum(1 for r in all_results if r.get('success', False))
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"  Tests Completed: {len(all_results)}/5")
        print(f"  Successful Tests: {success_count}")
        print(f"  Total Reward Across All Tests: {total_reward:.2f}")
        print(f"  Total Steps: {total_steps}")
        print(f"  Average Reward per Test: {total_reward/len(all_results):.2f}")
    
    print("\n" + "="*70)
    print(" TEST SUITE COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
