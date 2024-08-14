from utils.gpt import GPT, GPTConfig
from algorithm_distillation import create_data_pool, train_model, evaluate_ad

if __name__ == '__main__':
    config = GPTConfig(
        block_size = 1024,
        obs_dim = 81,
        act_dim = 5,
        vocab_size = 1,
        n_layer = 12,
        n_head = 12,
        n_embd = 768,
        dropout = 0.0,
        bias = True
    )
    model = GPT(config)

    taskpool = create_data_pool(
        num_tasks=64,
        num_episodes=1000,
        max_epsiode_len=20,
        gamma=0.99
    )
    losses = train_model(
        model=model,
        taskpool=taskpool,
        steps=1000,
        history_sample_len=50
    )

    done = evaluate_ad(
        model=model,
        taskpool=taskpool,
        max_steps=20,
        log=True,
        goal_from_tasks=False
    )