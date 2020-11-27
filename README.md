### Text Generation

Using deep learning model to generate Harry Potter-liked stories . Given a sequence of words as seed, this model predicts the following words.

#### Pre-process the text data

1. replace '--' with a space ' '.
2. tokenize the text by whits space
3. remove puntuation from each token
4. remove all tokens that are not alphabetic
5. convert to lower case

#### Word embedding

* Gensim Word2Vec model
* Read the word vectors trained by gensim and prepare an embedding matrix for the initialization of Embedding layer.

#### LSTM deep learning model

Define a deep learning sequential model with one embedding layer initialized by the word vectors pre-trained by gensim, two LSTM layers with 256 memory cells, one dropout layer with dropout rate 0.2, one dense layer with 256 neurons and relu activation, another dropout layer with dropout rate 0.2, and one output dense layer with 7731 neurons and softmax activation.

#### Stochastic sampling

Stochastic sampling introduces randomness in the sampling process, by sampling from the probability distribution for the next word. In order to control the amount of stochasticity in the sampling process, a parameter called the softmax temperature that characterizes the entropy of the probability distribution is introduced. Given a temperature value, a new probability distribution is computed from the predicted probability distribution of the model. Higher temperature results in higher entropy of the new probability distribution and generates more creative text data. On the contrary, lower temperature results in less randomness of the distribution and generates much more predictable data.

#### Generated stories

Story: 1
Seed Text: mr dursley might have been drifting into an uneasy sleep

Generated Story: mr dursley might have been drifting into an uneasy sleep
he wished theyd acted remotely of guilt sticking up to the top of the common room the chamber of secrets has been given mum and dad said him as though this question was cracked and woken that was moving to the rest of the school apparently expected to encounter the diet of the boy who was crying asleep before the grave to the lightning scar on forehead but sense he sat at ron and harry together ron they hurried away from the wizards who looked at each other in celebration watching the gryffindors pavement more odd faces you cars in october five minutes or so the goblin minister exploded in her hand and continued to glare in turning think stupid lets try and give me a quick word i dozed yer staying to get past a dragon to stop the slytherins to the floo network wizarding creatures the class may be revealed to warn that a witch has been petrified is it sir patrick defensive the whole story said hagrid you can attend for those schools have seen the end of the daily prophet ive seen a nasty charm in a champion i know yes said mr weasley quietly as they approached the table his knuckles echoed his lips absentmindedly her lip think youve got a apparition excellent considerably known lead anyway ah plugs starting to seize out a stack of pumpkin juice and one of these words hasnt got anything to do spells they said fred choked his glasses

Story: 2
Seed Text: harry and hermione walked on the street

Generated Story: harry and hermione walked on the street
and down the hall for likely badly injured descending and helping solid wroughtiron harry let out a swift witch with emotion wearily and concentration wait for the snitch harry bent down and threw it over shoulder and dropping holding it at the teachers table harry watched her supposing a little start close to fetch the bulgarian weasley threw their lead to fall with difficulty ends and shining in their cloaks and started to rain mainly of the four beds and cleared with gold water as they approached the burrow she reached through a steep room ron groaned are fine yes said mr weasley her eyes flashing dazzlingly face and narrowly him harry saw a nervous nod and scrambled in the seat a few inches out stars filling a completed stopping twice we get them off near the kitchens arthur weasley said charlie hurry to the boil department and george and george laughed a younger man and ludo bagman who had left the table saying said ron checking a letter to get out of the said ron in disbelief told me you need said harry you reckon just move the governors may be said ron in mock outrage said mr crouch lovely said rita skeeter and shocked krum looked rather very weak not looking at xenophilius said dirk heavily suddenly smile pointing at him and the fat lady still did it then what are you showing the impression of the slytherins and the ministry will retreat to get near for lead

Story: 3
Seed Text: ron joined the gryffindor quidditch team and

Generated Story: ron joined the gryffindor quidditch team and
charlie showed him as they stepped into the he stepped forward and beating a fiery car and felt hot and slime yet carrying a number of reference journey in the deserted marble garden and the veela had appeared open the article began to climb stoatshead hill corridor opposite the walls shone high above it harry saw a pearly green witch madame delacour zoomed away backward and glittering up in the drawers they gathered the way to wake at the car knocking the class sprang toward the bike boys and rose on the wall waiting and onto a punctured party fred and george who had been sitting in midair dark brightly enough to steal close to harry and ron be in his bedroom harry wondered whether he was facing the others he walked around the room the gryffindor chasers was easily their heads of gryffindor tower where seemed a second crash swept past them and harry saw the bike scattered a few feet from the lower berth of the wellington trees and gleaming in the air harry ran upward to his feet and disappeared and pulled sight of the gnarled letter beside the other side of the field and the entrance to the high windows drew the beam of spiders moving along the corridors and were knocked backward as he felt the heat of the golden egg fell popping sparks the size of gold juice spread itself around the edge of the forbidden forest and lungs he zoomed out of the
