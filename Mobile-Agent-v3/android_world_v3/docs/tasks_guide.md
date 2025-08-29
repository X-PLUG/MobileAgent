# Adding new tasks in AndroidWorld

This section provides a step-by-step guide to extending tasks in AndroidWorld, focusing on the two most common ways Android applications store data: SQLite databases and the file system. We use *Simple Calendar Pro* as an illustrative example for SQLite, and *Markor* as an example for file system storage. This guide will cover:

1. Determining how an app stores its data
2. Exploring an app's internal structure
3. Creating and validating a new task using SQLite
4. Creating and validating a new task using file system storage

## 1. Determining how an app stores its data

Before extending a task in AndroidWorld, it is essential to determine how the app in question stores its data. This guide focuses on SQLite and file system storage, but the principles can be adapted for alternative storage systems like SharedPreferences.

To identify whether an app uses SQLite or the file system:

1. **Access the app's data directory:** Use the following ADB command to navigate to the app’s data directory:
    ```bash
    adb shell ls data/data/<package_name>/
    ```
    This command lists the files and directories within the app's data directory. Replace `<package_name>` with the app’s package name (e.g., `com.simplemobiletools.calendar.pro`).

2. **Check for SQLite database files:** Within the app’s data directory, look for a `databases` folder, which typically contains SQLite database files. For example:
    ```bash
    adb shell ls data/data/com.simplemobiletools.calendar.pro/databases/
    ```
    If you find `.db` files, the app likely uses SQLite for data storage.

3. **Check for file system storage:** If there is no `databases` folder, or if the app stores data outside of the database, look for a `files` directory or other folders containing data files (e.g., text files, images). For example:
    ```bash
    adb shell ls data/data/<package_name>/files/
    ```
    If you find files such as `.txt`, `.json`, or other custom file types, the app is likely using the file system for data storage.

## 2. Exploring an app's internal structure

Once you have identified that an app uses SQLite or file system storage, the next step is to explore the database schema or file contents. This information is used for creating a new task that interacts with the app’s data.

### Exploring SQLite databases

1. **View the database schema:** Use the following command to examine the schema of a specific table in the SQLite database. The schema provides the structure of the table, including the columns and their data types:
    ```bash
    adb shell "sqlite3 data/data/com.simplemobiletools.calendar.pro/databases/events.db '.schema Events'"
    ```
    This command returns the schema for the `Events` table, showing the structure of calendar events stored by the app. The schema output will list the columns (e.g., `start_ts`, `end_ts`, `title`) and their respective data types (e.g., `INTEGER`, `TEXT`).

2. **Query the database:** You can retrieve data from the SQLite database to understand the types of records stored in it. For example, to view all records in the `events` table:
    ```bash
    adb shell "sqlite3 data/data/com.simplemobiletools.calendar.pro/databases/events.db 'SELECT * FROM events;'"
    ```
    This command outputs the contents of the `events` table, allowing you to inspect actual data entries and understand how they relate to the task you are developing.

### Exploring file system storage

1. **Inspect file contents:** To understand how data is stored in files, you can use ADB commands to view the contents of specific files. For example, to view a text file:
    ```bash
    adb shell cat data/data/<package_name>/files/<file_name>.txt
    ```
    This command outputs the contents of the specified text file, allowing you to understand the data structure and format used by the app.

2. **Download and explore files locally:** If you need to perform more complex analysis, you can pull files from the device to your local machine for inspection. For example:
    ```bash
    adb pull data/data/<package_name>/files/<file_name>.txt /local/directory/
    ```
    This command copies the specified file to your local machine, where you can use tools such as text editors or JSON viewers to explore the file’s contents.

## 3. Creating and validating a new task using SQLite

With the schema and data in hand, you can now create a new task in AndroidWorld. A key advantage of AndroidWorld is the use of abstractions that simplify task creation. For example below we use the `sqlite_validators.AddMultipleRows` class and `sqlite_validators.validate_rows_addition_integrity` functions, which encapsulate the key logic for interacting with SQLite databases and verifying that new items have been added. This allows developers to focus on defining the specifics of the task without needing to manually handle the intricacies of database operations. Below is a step-by-step guide to extending a task for an app using an SQLite database.

1. **Define the data class:** Start by defining a data class that mirrors the structure of the table you are working with. This class will represent the data rows in Python. For example, for the `Events` table:
    ```python
    @dataclasses.dataclass(frozen=True)
    class CalendarEvent:
      start_ts: int
      end_ts: int
      title: str
      location: str = ''
      description: str = ''
      repeat_interval: int = 0
      repeat_rule: int = 0
    ```
    This class captures the relevant fields from the `Events` table, providing a structured way to handle data within AndroidWorld.

2. **Create a base task class:** Develop a base class that handles common logic for interacting with the SQLite database. This includes specifying the database path, table name, and any necessary validation logic:
    ```python
    class _SimpleCalendar(sqlite_validators.SQLiteApp):
      """Base class for calendar tasks and evaluation logic."""

      app_name_with_db = "simple calendar pro"
      app_names = ("simple calendar pro",)

      db_key = "id"
      db_path = "data/data/com.simplemobiletools.calendar.pro/databases/events.db"
      table_name = "events"
      row_type = CalendarEvent
    ```
    This base class provides a foundation for interacting with the app's SQLite database, specifying key details like the database path and table name. By leveraging `sqlite_validators.AddMultipleRows`, you can define the task and its associated logic with minimal additional code:

3. **Implement task logic:** Create a task-specific class that extends the base class. This class should define the task template, generate parameters, and validate the task:
    ```python
    class SimpleCalendarAddOneEvent(sqlite_validators.AddMultipleRows, _SimpleCalendar):
      """Task for creating a calendar event in Simple Calendar Pro."""

      complexity = 2
      template = (
          "In Simple Calendar Pro, create a calendar event on {year}-{month}-{day}"
          " at {hour}h with the title `{event_title}' and the description"
          " `{event_description}'. The event should last for {duration_mins} mins."
      )

      @classmethod
      def _get_random_target_row(cls) -> CalendarEvent:
        """Generates a random calendar event."""
        return events_generator.generate_event(
            datetime_utils.create_random_october_2023_unix_ts()
        )

      def validate_addition_integrity(
          self,
          before: list[CalendarEvent],
          after: list[CalendarEvent],
          reference_rows: list[CalendarEvent],
      ) -> bool:
        """Validates the integrity of the event addition."""
       return sqlite_validators.validate_rows_addition_integrity(
          before, after, reference_rows,
          compare_fields=[
          'start_ts',
          'end_ts',
          'title',
          'location',
          'description'
      ]
      )

      @classmethod
      def generate_random_params(cls) -> dict[str, Any]:
        """Generate random parameters for a new calendar event task."""
        event = cls._get_random_target_row()
        n_noise_events = random.randint(0, 20)
        return {
           'year': device_constants.DT.year,
           'month': device_constants.DT.month,
           'day': event.start_datetime.day,
           'hour': event.start_datetime.hour,
           'duration_mins': event.duration_mins,
           'event_title': event.title,
           'event_description': event.description,
            sqlite_validators.ROW_OBJECTS: [event],
            sqlite_validators.NOISE_ROW_OBJECTS: generate_noise_events(
                [event], n_noise_events
            )
        }
    ```
    This class defines a specific task (adding a calendar event) and includes the logic for generating task parameters, validating task execution, and managing noise events.

4. **Integrate and test:** After implementing the task, manually test the validation logic to ensure it behaves as expected. AndroidWorld will automatically clear the database and all app state when `initialize_state` and/or `tear_down` are called.

## 4. Creating and validating a new task using file system storage

For apps that use the file system for data storage, you can extend AndroidWorld by creating tasks that interact with files. Below is a step-by-step guide using the *Markor* app as an example.

1. **Define the task class:** Create a class that defines the task of interacting with files. This might include creating, deleting, or modifying files. For example, to create a new note in *Markor*:
    ```python
    class MarkorCreateNote(task_eval.TaskEval):

      app_names = ("markor",)
      complexity = 2
      schema = file_validators.CreateFile.schema
      template = (
          "Create a new note in Markor named {file_name} with the following text:"
          " {text}"
      )

      def __init__(self, params: dict[str, Any]):
        """See base class."""
        super().__init__(params)

        self.create_file_task = file_validators.CreateFile(
            params, device_constants.MARKOR_DATA
        )

      def initialize_task(self, env: interface.AsyncEnv) -> None:
        super().initialize_task(env)
        self.create_file_task.initialize_task(env)

      def is_successful(self, env: interface.AsyncEnv) -> float:
        super().is_successful(env)
        return self.create_file_task.is_successful(env)

      @classmethod
      def generate_random_params(cls) -> dict[str, str | int]:
        return {"file_name": _generate_random_file_name(), "text": _generate_random_file_text()}

      def tear_down(self, env: interface.AsyncEnv) -> None:
        super().tear_down(env)
        self.create_file_task.tear_down(env)
    ```
    This class defines a task for creating a new note in the *Markor* app and uses the `CreateFile` validator to check the successful creation of the note file.

2. **Validate and test:** Similar to SQLite tasks, test the file interaction logic to ensure it behaves correctly.

## Conclusion

By following this guide, developers can extend AndroidWorld to support new tasks for various apps that use SQLite databases or the file system. The process involves determining how an app stores its data, exploring the app’s internal structure, and then creating a task using the existing evaluation logic. By leveraging these common building blocks, developers can extend AndroidWorld to a larger set of applications and tasks.
