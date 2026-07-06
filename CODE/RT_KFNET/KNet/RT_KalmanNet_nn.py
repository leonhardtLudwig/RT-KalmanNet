# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 09:42:01 2025
In this file we build the neural network presented at pag. 5 in the presentation's slide
It is composed by 
1) a fully connected layer with two layer
2) a DNN with feedback
"""

# ==============================================================================
# TODO LIST: REFACTORING RETE NEURALE (TASK 2 - Passaggio a GRU)
# ==============================================================================
# 
# ✅ COSA FARE (Aggiungere/Modificare):
# [x] Sostituire l'architettura core: Rimuovere il ciclo `for` e `nn.ModuleList()`
#     e inserire al suo posto una singola `nn.GRU(..., batch_first=True)`.
# [x] Aggiornare il costruttore (__init__): Deve ricevere parametri adatti alla GRU
#     (es. `input_size`, `gru_hidden_size`) al posto della vecchia lista di layer.
# [x] Modificare la firma del forward: Deve diventare `def forward(self, x, h_t=None):`
#     per accettare l'input corrente e la memoria del passo precedente.
# [x] Modificare il return: Il forward deve restituire la tupla `(c_t, h_t)`.
# [x] Cambiare l'attivazione finale: Sostituire `torch.sigmoid` con 
#     `torch.nn.functional.softplus` per permettere a `c` di superare il valore 1.
# 
# 🚫 COSA NON FARE (Rimuovere assolutamente):
# [ ] ELIMINARE `self.previous_output = torch.zeros(...)` dal costruttore. La 
#     memoria non deve risiedere fissa nella classe.
# [ ] ELIMINARE `final_output.clone().detach().requires_grad_(False)`. Questo
#     comando distrugge i gradienti temporali e rende impossibile la BPTT (Task 1).
# [ ] ELIMINARE i vari `torch.reshape`, `torch.squeeze` e `torch.cat` nel forward. 
#     La GRU gestisce l'unione di input e memoria internamente.
# 
# 🧠 NOTA CONCETTUALE:
# Mantieni il livello lineare iniziale (`self.fcl`) per estrarre le feature e 
# il livello finale (`self.output_layer`) per comprimere l'uscita della GRU 
# in un singolo scalare (la tolleranza c).
# ==============================================================================

import torch
import torch.nn as nn


class RT_KalmanNet_nn(nn.Module):
    def __init__(self, input_size_fcl, gru_hidden_size):
        """
        Args:
            input_size (int): Numero di neuroni in ingresso al Fully Connected Layer.
            gru_hidden_size (int): Numero di neuroni per l'espansione e per la memoria della GRU.
        """
        super().__init__()

        #Creation of the fully connected layer
        self.fcl_in = nn.Linear(input_size_fcl, gru_hidden_size)

        self.gru = nn.GRU(
            input_size = gru_hidden_size,
            hidden_size = gru_hidden_size,
            batch_first = True
        )

        self.fcl_out = nn.Linear(gru_hidden_size, 1)
        


    def forward(self, x, h_t = None):
        """
        Args:
            x (Tensor): input of the network (expected row vector of dimension input_size_fcl)
            h_t (Tensor, opzionale): La memoria (hidden state) dell'istante precedente.

        Returns:
            final_output (Tensor): Il parametro scalare c_t.
            h_t_nuovo (Tensor): La memoria aggiornata da passare all'istante successivo.
        """
        # Fully Connected Layer
        #x_feat = torch.relu(self.fcl_in(x))
    
        x_feat = self.fcl_in(x)  

        # --- GESTIONE DIMENSIONI PER LA GRU ---
        # Poiché stiamo processando un istante alla volta dentro il ciclo for, 
        # forziamo il tensore ad avere 3 dimensioni: (Batch=1, Sequenza=1, Features)
        # Se x_feat è [gru_hidden_size], diventa [1, 1, gru_hidden_size]
        if x_feat.dim() == 1:
            x_feat = x_feat.unsqueeze(0).unsqueeze(0)
        elif x_feat.dim() == 2:
            x_feat = x_feat.unsqueeze(1)

        # 2. IL CORE: Passaggio nella GRU
        # Diamo in pasto le feature e lo stato precedente.
        # out_gru è l'output per questo istante, h_t_nuovo è la memoria aggiornata.
        out_gru, h_t_nuovo = self.gru(x_feat, h_t)

        # 3. Output Layer
        # Passiamo l'output della GRU nell'ultimo livello lineare
        # c_raw passa da gru_hidden_size a 1
        c_raw = self.fcl_out(out_gru)

        # 4. Attivazione e pulizia
        # Rimuoviamo le dimensioni extra create per la GRU (il batch e la sequenza 1)
        c_raw = c_raw.squeeze()
        
        # Garantiamo che la tolleranza sia positiva (>0)
        final_output = torch.nn.functional.softplus(c_raw)

        return final_output, h_t_nuovo
