# Pattern_Recognition

    -> Trabalhos desenvolvidos a partir desse código, citar o paper disponível.
        -> Os experimentos citados no artigo encontra-se no arquivo experimento.py
    

    Aqui são construidas as classes do ELM para classificação
    São consideradas duas estruturas: 
        ELMClassifier: ELM padrão com uma camada oculta
        ELMMLPClassifier: ELM com Multicamadas (funcionando parcialmente)
    A classe BaseELM rege todo o funcionamento do ELM e funciona como uma superclasse para as outras duas citadas acima. A estrutura é como a seguir:
        
                            BaseELM
                                '
                     -----------------------
                     '                     '
                 ELMClassifier      ELMMLPClassifier
    
    O parâmetro func_hidden_layer recebe uma função para os pesos entre a entrada e a camada oculta. Todas as funções possíveis se encontram em aux_function.py, onde qualquer outra função para os pesos podem ser implementados. 
    O parâmetro activation_func controla o tipo de função de ativação utilizado. As implementações também estão no arquivo aux_func.py
    O parâmetro regressor controla qual o tipo de regressor será utilizado no processo. Existem 3 opções implementadas:
        pinv: pseudoinversa padrão do ELM,
        ls_reg: regressão pelo LS primal (padrão), mas os resultados mostraram que o resultado é equivalente ao método da pseudoinversa.
        ls_dual: regressão pelo LS dual. Este tem 2 parâmetros:
            lbd: termo de regularização
            degree: grau do kernel polinomial
            sparse: resolução com matriz esparsa. Mantenha falso, pois a implementação deste método não sucedeu-se muito bem. A tentativa era usar gradiente conjugado na resolução.
            
