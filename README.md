## Seamless Cloning | Numerik (WS 25/26)

Implementation of [Poisson Image Editing](https://en.wikipedia.org/wiki/Gradient-domain_image_processing) in Python.
Dependencies include *numpy* and *scipy* for conjugate gradients, *scikit-image* for file-reading and *matplotlib* for
visualization.

1. Open the directory in a unix-terminal with bash and create a **virtual environment**. For fish users, add `activate.fish` when sourcing.

    ```Bash
    python3 -m venv venv
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```
    
2. Check for **dependencies** by running

    ```Bash
    pip install -r requirements.txt
    ```
    
3. **Execute** the script

    ```Bash
    python3 main.py
    ```

Keep in mind that you have to close every plot manually if you run from the command line. If that sounds tedious, use an IDE with matplotlib support ;)
