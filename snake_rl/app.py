import streamlit as st
import subprocess
import os
import torch
from pathlib import Path
import pygame
import sys
import threading
import time
from PIL import Image
import io
import numpy as np

pygame.init()
pygame.font.init()

st.set_page_config(
    page_title="Snake Game AI Training & Testing Platform",
    page_icon="🐍",
    layout="wide"
)



def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        shell=True
    )
    
    # 创建一个空的输出容器
    output_container = st.empty()
    output_text = ""
    
    # 实时读取输出
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            output_text += output
            output_container.text(output_text)
    
    return process.poll()

def capture_pygame_surface(surface):
    data = pygame.image.tostring(surface, 'RGB')
    image = Image.frombytes('RGB', surface.get_size(), data)
    return image

def run_game(algorithm, model_path, game_container, stop_event):
    """Run game in Streamlit"""
    # Initialize pygame properly
    pygame.init()
    pygame.font.init()
    
    # Initialize game
    from snakeq_rl.snake_game import SnakeGame
    env = SnakeGame()
    
    # Create agent
    if algorithm == "PPO":
        from snakeq_rl.ppo.agent import PPOAgent
        agent = PPOAgent(28, 4, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    else:  # DQN
        from snakeq_rl.dqn.agent import DQNAgent
        agent = DQNAgent(28, 4, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Load model
    if model_path:
        agent.load(model_path)
    
    # Game loop
    running = True
    episode = 0
    best_score = 0
    
    try:
        while running and not stop_event.is_set():
            # Reset game
            env.reset()
            state = env.get_state_representation()
            episode += 1
            steps = 0
            score = 0
            
            while running and not stop_event.is_set():
                # Get action
                if algorithm == "DQN":
                    # Convert state to numpy array if it's not already
                    if isinstance(state, str):
                        # If state is a string, we need to parse it into a numerical array
                        # This is a temporary fix - the get_state_representation method should be updated
                        state = np.zeros(28)  # Use a default state for now
                    action = agent.select_action(state, epsilon=0.0)
                else:
                    action = agent.select_action(state, epsilon=0.0)
                
                move = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
                
                # Make move
                alive, apple_eaten = env.make_move(move)
                steps += 1
                if apple_eaten:
                    score += 1
                
                # Update display
                env.draw()
                screen = env.window.screen
                
                # Convert Pygame surface to PIL Image
                data = pygame.image.tostring(screen, 'RGB')
                image = Image.frombytes('RGB', screen.get_size(), data)
                
                # Display in Streamlit
                game_container.image(
                    image, 
                    caption=f"Episode {episode} - Score: {score} - Steps: {steps}"
                )
                
                # Check if game over
                if not alive:
                    if score > best_score:
                        best_score = score
                    break
                
                # Add small delay
                time.sleep(0.1)
    finally:
        # Clean up pygame
        pygame.quit()

def main():
    try:
        st.title("🐍 Snake Game AI Training & Testing Platform")
        
        # Create sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio("Select Function", ["Training", "Testing"])
        
        if page == "Training":
            st.header("Model Training")
            
            # Select algorithm
            algorithm = st.selectbox("Select Algorithm", ["PPO", "DQN"])
            
            # Training parameters
            st.subheader("Training Parameters")
            col1, col2 = st.columns(2)
            
            with col1:
                max_episodes = st.number_input("Max Episodes", value=100000, step=1000)
                max_steps = st.number_input("Max Steps per Episode", value=200, step=10)
                epsilon = st.number_input("Initial Exploration Rate", value=1.0, step=0.1)
            
            with col2:
                epsilon_decay = st.number_input("Exploration Rate Decay", value=0.9999, step=0.0001)
                epsilon_min = st.number_input("Min Exploration Rate", value=0.01, step=0.01)
                batch_size = st.number_input("Batch Size", value=64, step=8)
            
            # Start training button
            if st.button("Start Training"):
                st.info(f"Starting {algorithm} training...")
                
                # Create model save directory
                model_dir = Path(f"snakeq_rl/model/{algorithm.lower()}")
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Run training command
                command = f"python -m snakeq_rl.{algorithm.lower()}.train"
                run_command(command)
                
                st.success("Training completed!")
        
        else:  # Testing page
            st.header("Model Testing")

            # 选择算法
            algorithm = st.selectbox("Select Algorithm", ["PPO", "DQN"])

            # 选择模型文件
            model_dir = Path(f"snakeq_rl/model/{algorithm.lower()}")
            if model_dir.exists():
                model_files = list(model_dir.glob("*.pth"))
                if model_files:
                    model_files = [str(f) for f in model_files]
                    selected_model = st.selectbox("Select Model File", model_files)

                                        # 初始化 session_state
                    if (
                        "game" not in st.session_state or
                        st.session_state.get("algorithm") != algorithm or
                        st.session_state.get("model_path") != selected_model
                    ):
                        # 确保 pygame 已初始化
                        pygame.init()
                        pygame.font.init()
                        from snakeq_rl.snake_game import SnakeGame
                        st.session_state.game = SnakeGame()
                        st.session_state.algorithm = algorithm
                        st.session_state.model_path = selected_model
                        st.session_state.score = 0
                        st.session_state.steps = 0
                        st.session_state.done = False
                        st.session_state.last_score_time = time.time()
                        st.session_state.last_score = 0
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        if algorithm == "PPO":
                            from snakeq_rl.ppo.agent import PPOAgent
                            st.session_state.agent = PPOAgent(28, 4, device)
                        else:
                            from snakeq_rl.dqn.agent import DQNAgent
                            st.session_state.agent = DQNAgent(28, 4, device)
                        st.session_state.agent.load(selected_model)
                        st.session_state.state = st.session_state.game.get_state_representation()
                        # 立即绘制初始画面
                        st.session_state.game.draw()

                    # 在页面顶部定义一个空的图像容器
                    image_placeholder = st.empty()

                    # 单步推进
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("下一步", key="step"):
                            if not st.session_state.done:
                                state = st.session_state.state
                                if algorithm == "DQN" and isinstance(state, str):
                                    state = np.zeros(28)
                                action = st.session_state.agent.select_action(state, epsilon=0.0)
                                move = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
                                alive, apple_eaten = st.session_state.game.make_move(move)
                                st.session_state.game.draw()  # 确保画面更新
                                st.session_state.state = st.session_state.game.get_state_representation()
                                st.session_state.steps += 1
                                if apple_eaten:
                                    st.session_state.score += 1
                                    st.session_state.last_score = st.session_state.score
                                    st.session_state.last_score_time = time.time()
                                if not alive or (time.time() - st.session_state.last_score_time > 30 and st.session_state.score == st.session_state.last_score):
                                    st.info(f"游戏结束! 得分: {st.session_state.score}, 步数: {st.session_state.steps}")
                                    st.session_state.done = True
                    with col2:
                        if st.button("重置游戏", key="reset"):
                            for k in ["game", "agent", "algorithm", "model_path", "score", "steps", "done", "state", "auto", "last_score_time", "last_score"]:
                                if k in st.session_state:
                                    del st.session_state[k]
                            st.rerun()

                    def set_auto():
                        st.session_state.is_auto_running = True

                    with col3:
                        st.button("自动运行", key="auto", on_click=set_auto)

                    # 自动运行功能
                    if st.session_state.get("is_auto_running", False) and not st.session_state.done:
                        state = st.session_state.state
                        if algorithm == "DQN" and isinstance(state, str):
                            state = np.zeros(28)
                        action = st.session_state.agent.select_action(state, epsilon=0.0)
                        move = ['UP', 'DOWN', 'LEFT', 'RIGHT'][action]
                        alive, apple_eaten = st.session_state.game.make_move(move)
                        st.session_state.game.draw()  # 确保画面更新
                        st.session_state.state = st.session_state.game.get_state_representation()
                        st.session_state.steps += 1
                        if apple_eaten:
                            st.session_state.score += 1
                            st.session_state.last_score = st.session_state.score
                            st.session_state.last_score_time = time.time()
                        if not alive or (time.time() - st.session_state.last_score_time > 30 and st.session_state.score == st.session_state.last_score):
                            st.info(f"游戏结束! 得分: {st.session_state.score}, 步数: {st.session_state.steps}")
                            st.session_state.done = True
                            if "is_auto_running" in st.session_state:
                                st.session_state.is_auto_running = False
                        time.sleep(0.1)  # 延迟 0.1 秒，展示每一步
                        # 动态更新图像
                        image_placeholder.image(
                            Image.frombytes('RGB', st.session_state.game.window.screen.get_size(),
                                           pygame.image.tostring(st.session_state.game.window.screen, 'RGB')),
                            caption=f"Score: {st.session_state.score} - Steps: {st.session_state.steps}"
                        )
                        if st.session_state.done:
                            st.info("Game Over!")
                            if "is_auto_running" in st.session_state:
                                st.session_state.is_auto_running = False
                else:
                    st.warning(f"No model files found in {model_dir}, please train a model first.")
            else:
                st.warning(f"Model directory {model_dir} does not exist, please train a model first.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all dependencies are installed correctly.")

if __name__ == "__main__":
    main()