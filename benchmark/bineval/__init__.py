"""bineval: pass/fail evaluation of a reader model over a long chat.

``arms`` builds the context each arm shows the reader, ``run_reader`` answers
the questions with that context (optionally with mass injection), and
``score_binary`` scores the answers.  ``build_cd_offline`` builds the
correlation diagram (CD) that the ``cd_*`` arms serialize.

Importing the package does no I/O and loads no model.
"""
