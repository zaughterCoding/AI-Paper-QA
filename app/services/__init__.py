"""Business services.

One method call is one complete business action ("ingest a document", "answer a
question"). A service validates business rules, coordinates several repositories, and
decides where transactions begin and end.

The API layer calls it, scripts call it, and evaluation code will call it: there is only
one copy of the business logic.
"""
