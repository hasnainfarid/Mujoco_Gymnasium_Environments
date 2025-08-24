#!/usr/bin/env python3
"""Simple test script for Quadruped Parkour Environment."""

import sys
import os
import time
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parkour_env import QuadrupedParkourEnv

def create_environment(render_mode='human'):
    """Create parkour environment."""
    print("🤖 Creating Quadruped Parkour Environment...")
    
    try:
        env = QuadrupedParkourEnv(render_mode=render_mode)
        print("✅ Environment created successfully!")
        return env
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return None

def test_environment_basic(env, episodes=5):
    """Test basic environment functionality."""
    print(f"\n🧪 Testing basic functionality ({episodes} episodes)...")
    
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}:")
        
        # Reset environment
        obs, info = env.reset()
        print(f"  ✅ Reset successful, obs shape: {obs.shape}")
        
        if env.render_mode == "human":
            print("  🎮 3D window should be visible now!")
            time.sleep(2)
        
        # Run episode
        total_reward = 0
        for step in range(100):
            # Simple trot gait
            action = generate_trot_gait(step * 0.01)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 20 == 0:
                progress = info.get('course_completion', 0.0)
                print(f"    Step {step}: reward={reward:.2f}, progress={progress:.1%}")
            
            if terminated or truncated:
                print(f"    Episode ended at step {step}")
                break
            
            time.sleep(0.02)  # Small delay for visualization
        
        print(f"  📊 Total reward: {total_reward:.2f}")
        print(f"  🏁 Final progress: {info.get('course_completion', 0.0):.1%}")

def generate_trot_gait(t):
    """Generate simple trot gait pattern."""
    action = np.zeros(16)
    
    # Trot gait: diagonal legs move together
    freq = 2.0  # Hz
    phase = 2 * np.pi * freq * t
    
    # Front left and back right (diagonal pair 1)
    fl_phase = phase
    br_phase = phase
    
    # Front right and back left (diagonal pair 2)
    fr_phase = phase + np.pi
    bl_phase = phase + np.pi
    
    # Hip flexion/extension
    action[1] = 20 * np.sin(fl_phase)    # FL hip flexion
    action[5] = 20 * np.sin(fr_phase)    # FR hip flexion
    action[9] = 20 * np.sin(bl_phase)    # BL hip flexion
    action[13] = 20 * np.sin(br_phase)   # BR hip flexion
    
    # Knee flexion
    action[2] = -30 * np.maximum(0, np.sin(fl_phase))    # FL knee
    action[6] = -30 * np.maximum(0, np.sin(fr_phase))    # FR knee
    action[10] = -30 * np.maximum(0, np.sin(bl_phase))   # BL knee
    action[14] = -30 * np.maximum(0, np.sin(br_phase))   # BR knee
    
    return action

def main():
    """Main function."""
    print("🤖🏃‍♂️ Quadruped Parkour Environment Test")
    print("=" * 50)
    
    # Create environment
    env = create_environment(render_mode='human')
    if env is None:
        return
    
    try:
        # Test basic functionality with more episodes
        test_environment_basic(env, episodes=20)
        
        print("\n🎉 All tests completed successfully!")
        
        if env.render_mode == "human":
            print("🎮 Visualization window will close in 3 seconds...")
            time.sleep(3)
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
    finally:
        env.close()
        print("✅ Environment closed")

if __name__ == "__main__":
    main()
