"""Data access layer.

Its only job is talking to the database: issuing SQL and turning rows into objects. It
does no validation, takes no business decisions and controls no transactions -- those
belong to the service layer.

Keeping the two apart matters because they change for different reasons. Swapping the
database, altering a table or tuning a query touches only this layer; changing a business
rule touches only the services. Mixed together, either kind of change drags in the rest.

Read methods are named ``get_*``/``list_*``/``count_*``, write methods ``create_*``/
``update_*``/``delete_*``.
"""
