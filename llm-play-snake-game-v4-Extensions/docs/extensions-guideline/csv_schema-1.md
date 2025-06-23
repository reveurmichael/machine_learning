Here's a clean, practical CSV schema for supervised learning on a 10x10 Snake game grid — balancing expressiveness, simplicity, and easy training:

---

# Recommended CSV Schema for Snake v0.03 (10x10 grid)

| Column Name        | Type     | Description                                                                            |
| ------------------ | -------- | -------------------------------------------------------------------------------------- |
| game\_id           | int      | Unique game session ID (for grouping, not used as feature)                             |
| step\_in\_game     | int      | Step number in the current game (optional, not used as input)                          |
| head\_x            | int      | Snake head X coordinate (0–9)                                                          |
| head\_y            | int      | Snake head Y coordinate (0–9)                                                          |
| apple\_x           | int      | Apple X coordinate (0–9)                                                               |
| apple\_y           | int      | Apple Y coordinate (0–9)                                                               |
| snake\_length      | int      | Current length of the snake                                                            |
| apple\_dir\_up     | int(0/1) | 1 if apple is above the snake head, else 0                                             |
| apple\_dir\_down   | int(0/1) | 1 if apple is below the snake head, else 0                                             |
| apple\_dir\_left   | int(0/1) | 1 if apple is left of the snake head, else 0                                           |
| apple\_dir\_right  | int(0/1) | 1 if apple is right of the snake head, else 0                                          |
| danger\_straight   | int(0/1) | 1 if immediate cell ahead is wall or snake body, else 0                                |
| danger\_left       | int(0/1) | 1 if immediate cell to the left of snake head direction is wall or snake body, else 0  |
| danger\_right      | int(0/1) | 1 if immediate cell to the right of snake head direction is wall or snake body, else 0 |
| free\_space\_up    | int      | Number of free squares in the up direction (0–9)                                       |
| free\_space\_down  | int      | Number of free squares in the down direction (0–9)                                     |
| free\_space\_left  | int      | Number of free squares in the left direction (0–9)                                     |
| free\_space\_right | int      | Number of free squares in the right direction (0–9)                                    |
| target\_move       | string   | Next move/action taken by the snake (one of: UP, DOWN, LEFT, RIGHT) — training label   |

---

### Explanation

* **game\_id, step\_in\_game:** Useful for bookkeeping or splitting data but **not inputs** to the model.
* **head\_x, head\_y, apple\_x, apple\_y:** Absolute positions, helpful if your model can learn spatial relations.
* **snake\_length:** Gives the model context about snake size (important for strategy).
* **apple\_dir\_**\*: Encodes the relative apple direction as four binary flags — easier for the model to interpret than raw coordinates.
* **danger\_**\*: Immediate danger flags indicate whether the snake would collide if it moved straight, left, or right relative to current heading.
* **free\_space\_**\*: Counts how many free squares exist in each cardinal direction before hitting a wall or snake body, indicating available maneuvering space.
* **target\_move:** The supervised label; the next direction chosen by the agent/player.

---

### Example CSV row (partial):

| game\_id | step\_in\_game | head\_x | head\_y | apple\_x | apple\_y | snake\_length | apple\_dir\_up | apple\_dir\_down | apple\_dir\_left | apple\_dir\_right | danger\_straight | danger\_left | danger\_right | free\_space\_up | free\_space\_down | free\_space\_left | free\_space\_right | target\_move |
| -------- | -------------- | ------- | ------- | -------- | -------- | ------------- | -------------- | ---------------- | ---------------- | ----------------- | ---------------- | ------------ | ------------- | --------------- | ----------------- | ----------------- | ------------------ | ------------ |
| 17       | 5              | 4       | 6       | 7        | 9        | 8             | 1              | 0                | 0                | 1                 | 0                | 1            | 0             | 3               | 2                 | 1                 | 5                  | UP           |
