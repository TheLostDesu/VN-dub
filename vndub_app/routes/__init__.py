"""HTTP routes of the desktop app (API.md section 3). Each module exposes ``router``."""

#: modules included by ``vndub_app.server.create_app`` (in this order)
MODULES = ("general", "jobs", "project", "translation", "casting", "run", "review", "processes")
