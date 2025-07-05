Coordinates range from (0,0) at the **bottom-left** to (GridSize-1,GridSize-1) at the **top-right**.

### Movement Example (Normal Move):
If S = `[HEAD (4,4), BODY1 (4,3)]` and you choose "LEFT":  
- new_head = (4-1, 4) = (3,4).  
- No collision if (3,4) is empty.  
- Insert old head (4,4) at front of BODY ⇒ BODY becomes `[(4,4), (4,3)]`.  
- Remove tail (4,3) (assuming not eating apple) ⇒ BODY becomes `[(4,4)]`.  
- Update HEAD to (3,4).  
- Resulting S = `[(3,4) (head), (4,4) (body1)]`.

## COORDINATE SYSTEM:
- "UP" means y+1  
- "DOWN" means y-1  
- "RIGHT" means x+1  
- "LEFT" means x-1  

Example Moves from (1,1):  
• UP → (1,2)  
• DOWN → (1,0)  
• RIGHT → (2,1)  
• LEFT → (0,1)


In our code, 
Snake_positions[0] is tail, [-1] is HEAD


