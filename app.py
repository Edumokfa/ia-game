import time
import numpy as np
import gymnasium as gym
import jogo_env
import pickle

def escolher_acao(estado, ambiente, tabela_q, epsilon):
    if np.random.rand() < epsilon:
        return ambiente.action_space.sample()
    else:
        return np.argmax(tabela_q[estado])

def treinar(episodios, ambiente, tabela_q):
    epsilon = 1.0

    for episodio in range(episodios):
        estado, _ = ambiente.reset()
        estado = tuple(estado)
        recompensa_total = 0
        
        for _ in range(100):
            acao = escolher_acao(estado, ambiente, tabela_q, epsilon)
            prox_estado, recompensa, concluido, _, _, = ambiente.step(acao)
            prox_estado = tuple(prox_estado)
            valor_q_atual = tabela_q[estado][acao]
            melhor_valor_q = np.max(tabela_q[prox_estado])
            novo_valor_q = valor_q_atual + 0.2 * (recompensa + 0.9 * melhor_valor_q - valor_q_atual) #isso faz alguma coisa legal que não sei explicar
            tabela_q[estado][acao] = novo_valor_q
            estado = prox_estado
            recompensa_total += recompensa
            
            if concluido:
                break
        
        if epsilon > 0.1:
            epsilon *= 0.995

    print(f"Recompensa do último treino: {recompensa_total}")


def executar(ambiente, tabela_q):
    estado, _ = ambiente.reset()
    estado = tuple(estado)
    recompensa_total = 0

    ambiente.render()
    for _ in range(100):
        acao = np.argmax(tabela_q[estado])
        prox_estado, recompensa, concluido, _, _ = ambiente.step(acao)
        prox_estado = tuple(prox_estado)
        ambiente.render()
        recompensa_total += recompensa
        estado = prox_estado
        
        time.sleep(1)

        if concluido:
            break

    print(f"Recompensa Total: {recompensa_total}")

if __name__ == '__main__':
    ambiente = gym.make(
        'jogo-sobrevivencia', 
        render_mode='human', 
        grid_rows=8, 
        grid_cols=8, 
        zombies_amount=8, 
        supplies_amount=8, 
        walls_amount=2, 
    )

    tamanho_observacao = ambiente.observation_space.nvec
    formato_estado = tuple(tamanho_observacao)
    tabela_q = np.zeros(formato_estado + (ambiente.action_space.n,))
    # Inicialmente treina com 1000 casos, depois executa um caso com o ambiente treinado e exibe visualmente.
    treinar(1000, ambiente, tabela_q)
    with open("solucao_tabela.pkl", "wb") as arquivo:
        pickle.dump(tabela_q, arquivo)
    
    with open('solucao_tabela.pkl', 'rb') as arquivo:
        tabela_q = pickle.load(arquivo)
    executar(ambiente, tabela_q)

    ambiente.close()
