import time
import numpy as np
import gymnasium as gym
import jogo_env
import pickle

taxa_aprendizado = 0.2
fator_desconto = 0.9
epsilon_minimo = 0.1
decrescimento_epsilon = 0.996
passos_maximos = 100

def escolher_acao(estado, ambiente, tabela_q, epsilon):
    if np.random.rand() < epsilon:
        return ambiente.action_space.sample()
    else:
        return np.argmax(tabela_q[estado])

def executar_q(episodios, ambiente, treinamento=True, renderizar=False):
    if treinamento:
        tamanho_observacao = ambiente.observation_space.nvec
        formato_estado = tuple(tamanho_observacao)
        tabela_q = np.zeros(formato_estado + (ambiente.action_space.n,))
    else:
        with open('solucao_tabela.pkl', 'rb') as arquivo:
            tabela_q = pickle.load(arquivo)

    if treinamento:
        treinar(episodios, ambiente, tabela_q)
    else:
        executar(ambiente, tabela_q)

    if treinamento:
        with open("solucao_tabela.pkl", "wb") as arquivo:
            pickle.dump(tabela_q, arquivo)

def treinar(episodios, ambiente, tabela_q):
    epsilon = 1.0

    for episodio in range(episodios):
        estado, _ = ambiente.reset()
        estado = tuple(estado)
        recompensa_total = 0
        
        for _ in range(passos_maximos):
            acao = escolher_acao(estado, ambiente, tabela_q, epsilon)
            prox_estado, recompensa, concluido, _, _, = ambiente.step(acao)
            prox_estado = tuple(prox_estado)
            
            valor_q_atual = tabela_q[estado][acao]
            melhor_valor_q = np.max(tabela_q[prox_estado])
            novo_valor_q = valor_q_atual + taxa_aprendizado * (recompensa + fator_desconto * melhor_valor_q - valor_q_atual)
            tabela_q[estado][acao] = novo_valor_q
            
            estado = prox_estado
            recompensa_total += recompensa
            
            if concluido:
                break
        
        if epsilon > epsilon_minimo:
            epsilon *= decrescimento_epsilon
        
        if episodio % 100 == 0:
            print(f"Episódio: {episodio}, Recompensa Total: {recompensa_total}")

def executar(ambiente, tabela_q):
    estado, _ = ambiente.reset()
    estado = tuple(estado)
    recompensa_total = 0

    ambiente.render()
    for _ in range(passos_maximos):
        acao = np.argmax(tabela_q[estado])
        prox_estado, recompensa, concluido, _, _ = ambiente.step(acao)
        prox_estado = tuple(prox_estado)
        ambiente.render()
        
        estado = prox_estado
        recompensa_total += recompensa
        
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

    executar_q(1000, ambiente, treinamento=True, renderizar=False)
    
    executar_q(1, ambiente, treinamento=False, renderizar=True)
    ambiente.close()
