# IMPORTANT NAMING CONVENTION
- Please name your model file as "ModelName.py" and the class as "ModelName". I.E. THE SAME NAME EXACTLY. 
- Also base your model class on the BaseModel class defined in BaseModel.py.

## Other info
Your model class must have a return_action method which takes in a state and returns a list of actions.

### State Dictionary Format
The state parameter passed to `return_action` is a dictionary with the following structure:

```python
{
    'did_crash': False,           # Boolean: whether the car has crashed
    'elapsed_ticks': 50,          # Integer: number of game ticks elapsed
    'distance': 567,              # Float: total distance traveled
    'velocity': {                 # Dictionary: current velocity
        'x': 10,                  # Float: velocity in x direction
        'y': 0                    # Float: velocity in y direction
    },
    'sensors': {                  # Dictionary: sensor readings (distances to obstacles)
        'front': 500,             # Distance straight ahead
        'right_front': 600,       # Distance to front-right diagonal
        'right_side': 700,        # Distance to right side
        'right_back': 800,        # Distance to back-right diagonal
        'back': 900,              # Distance straight back
        'left_back': 400,         # Distance to back-left diagonal
        'left_side': 300,         # Distance to left side
        'left_front': 200,        # Distance to front-left diagonal
        'left_side_front': 350,   # Distance to left-side-front
        'front_left_front': 450,  # Distance to front-left-front
        'front_right_front': 550, # Distance to front-right-front
        'right_side_front': 650,  # Distance to right-side-front
        'right_side_back': 750,   # Distance to right-side-back
        'back_right_back': 850,   # Distance to back-right-back
        'back_left_back': 950,    # Distance to back-left-back
        'left_side_back': 150     # Distance to left-side-back
    }
}
```
