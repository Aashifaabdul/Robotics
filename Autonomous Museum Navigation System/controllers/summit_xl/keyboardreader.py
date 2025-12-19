from controller import Keyboard

class KeyboardReader(object):
    """
    This class reads keys from the keyboard using the method 'get_command'
    and returns the text provided once the 'enter' key is pressed.
    """
    def __init__(self, ts):
        self.keyboard = Keyboard()
        self.keyboard.enable(ts)
        self.temp_text = ''
        self.last_key = -1
        self.mapping = {32:' ', 44:',', 65579:'_'}
        
        self.goals = ["duck", "chair", "dog", "phone", "cat", "computer", "flower", "ball", "gnome"]

    def get_command(self):
        """
        Returns a string after a command is provided. Otherwise it returns
        None.
        """
        k = self.keyboard.getKey()
        if k != -1:
            if k == self.last_key:
                return None
            self.last_key = k
            if k==4:
                ret = self.temp_text.lower()
                self.temp_text = ''
                if ret == 'bal':
                    ret = 'ball'
                if ret not in self.goals:
                    ret = "(unknown) " + ret
                return ret
            elif k==3:
                self.temp_text=self.temp_text[-1]
            elif k in self.mapping.keys():
                self.temp_text+=self.mapping[k]
            else:
                self.temp_text+=chr(k)
        return None
    