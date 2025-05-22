GMRA in Python

Instruction
1. The code will save data to datasets at root folder. It is best to symlink your datasets using ln -s /your_usual_place_to_store_data/ . (and also dont do stuff like rm -rf datasets/*)
2. It will run fine with python  3.8 and torch '2.4.1+cu121'
3. src contains only core code to gmra.
4. experiments contains code to perform experiments
5. logging contains loggings, graph, tensorboards, etc.