Ecco una versione completa e accuratamente ripulita della trascrizione audio presente nelle tue fonti. 

L'audio originale è in inglese e contiene numerosi errori dovuti alla trascrizione automatica (ad esempio, il software ha trascritto *“karma filter”* per Kalman filter, *“carmanate”* per KalmanNet, *“jar”* per GRU, *“say”* per il parametro $c$, e *“pass”* per Python). Ho corretto tutti i termini tecnici, rimosso le esitazioni del parlato e tradotto il tutto in un dialogo logico e scorrevole in italiano tra il Professore (che assegna il progetto) e gli Studenti.

***

### Transcript Ripulito: Progetto Robust EKF e KalmanNet

**Professore:** Immagino che conosciate già il Filtro di Kalman, giusto?
**Studente:** Sì.
**Professore:** Bene, quindi non c'è bisogno che vi introduca l'Extended Kalman Filter (EKF). Alla fine, questo filtro è fortemente basato sul modello (Model-Based). Avete bisogno di una conoscenza precisa del modello, inclusi i parametri delle funzioni della dinamica e la varianza del rumore. Nelle applicazioni reali, come ad esempio la navigazione di droni (UAV), questo rumore è sconosciuto e alcuni parametri del modello possono mancare. Inoltre, la discretizzazione dei modelli fisici a tempo continuo causa disallineamenti di modellazione. Questa incertezza fa sì che l'EKF abbia prestazioni scarse, e questo è il nostro punto di partenza.

**Studente:** Ok.
**Professore:** L'EKF standard usa lo Jacobiano per linearizzare il modello non lineare nei passaggi di predizione e filtraggio. Il filtro, fondamentalmente, calcola il "Guadagno di Kalman" per decidere se fidarsi maggiormente del modello dinamico o della misurazione.
Io ho proposto un nuovo "Robust EKF". Assumiamo che il modello reale sia sconosciuto, ma che si trovi all'interno di un set (una sorta di "palla" di tolleranza) attorno a un "modello nominale" semplice, come ad esempio un modello a velocità costante. L'unica differenza del mio algoritmo rispetto all'EKF standard è che aggiungiamo un nuovo parametro scalare (la tolleranza $c$) per ricalibrare la matrice di covarianza predetta. Questo dà robustezza al filtro anche quando il modello non è preciso. Tuttavia, nelle applicazioni reali non conosciamo a priori questa tolleranza $c$, perché varia nel tempo in base all'ambiente (ad esempio, se il segnale GPS degrada o viene perso). Il vostro progetto consisterà nel trovare un modo per stimare dinamicamente questo parametro.

**Professore:** Vi introduco un altro algoritmo: la KalmanNet. In questo approccio, si usa il deep learning per sostituire direttamente il blocco che calcola il Guadagno di Kalman.
**Studente:** Quindi non c'è la matrice P (covarianza) in questo calcolo?
**Professore:** Esatto, non c'è bisogno di calcolare le matrici di covarianza e non c'è bisogno di conoscere le statistiche del rumore ($Q$ e $R$). Si usa semplicemente una rete neurale per computare il guadagno, superando i problemi di non linearità e i limiti del modello esatto. È sicuramente un algoritmo computazionalmente più pesante, ma molto potente.

**Professore:** Quello che voglio che facciate è combinare questi due algoritmi. Il problema della KalmanNet classica è che il suo output (il Guadagno di Kalman) è una matrice molto grande e complessa da calcolare (es. per uno stato di dimensione 20). Invece, nel mio Robust EKF, il parametro di tolleranza $c$ che dobbiamo trovare è soltanto uno scalare. Voglio che usiate la struttura della rete neurale per stimare questo singolo parametro $c$ a ogni passo temporale.

**Professore:** Ho già completato l'80% del lavoro. Vi darò il codice per il problema di navigazione UAV, per l'algoritmo robusto e la struttura base della rete. Tuttavia, il mio codice attuale usa una vecchia e superata rete neurale densa (DNN). Voglio che la sostituiate con una GRU (Gated Recurrent Unit), la stessa usata nel paper della KalmanNet. Le GRU sono eccellenti per l'apprendimento di sequenze temporali perché riescono a mantenere memoria delle informazioni passate.

**Studente:** Dobbiamo implementare noi l'addestramento di questa rete?
**Professore:** Sì. Dovrete leggere attentamente la Sezione D del paper della KalmanNet per capire la strategia di training. Al momento, il mio codice usa una singola traiettoria per l'addestramento, il che non va bene per il deep learning. Dovrete implementare una strategia su mini-batch sfruttando molti dati per migliorare l'efficienza.
Quindi, riassumendo, dovrete:
1. Migliorare la mia strategia di training (basandovi sul paper KalmanNet).
2. Sostituire il core della rete neurale con una GRU.
3. Verificare se l'algoritmo funziona. Vi darò uno script che genera dati simulati in cui è noto il valore "ground truth" di $c$ (sia come valore costante, sia come parametro che varia lentamente nel tempo), così potrete confrontare le vostre stime con il valore reale.
4. Confrontare il vostro nuovo algoritmo con l'EKF standard, il mio Robust EKF e la KalmanNet pura. 

**Studente:** Dobbiamo aspettarci prestazioni migliori dal nostro?
**Professore:** Speriamo di sì! Spero che il vostro si classifichi al primo posto per prestazioni e al secondo per efficienza nel tempo di calcolo, ma dovrete valutarlo misurando anche i tempi di esecuzione.
**Studente:** Abbiamo il codice in Python?
**Professore:** Io ho usato MATLAB per alcune cose, ma troverete i codici in Python nel link allegato al paper della KalmanNet. L'importante è che sviluppiate tutti i metodi finali in Python per poter fare un confronto equo.

**Studente:** Qual è la scadenza per la consegna?
**Professore:** Non c'è una scadenza fissa. Potete completare il progetto in questo semestre, nel prossimo, o a settembre. Prendetevi il vostro tempo per fare un buon lavoro. Alla fine dovrete solo consegnare il codice e fare una presentazione (anche su Zoom) in cui spiegate cosa avete fatto e lo testiamo. Non è necessario scrivere alcun report, a meno che non vogliate chiarire passaggi complessi. Avrete anche a disposizione due incontri intermedi con me per fare domande.