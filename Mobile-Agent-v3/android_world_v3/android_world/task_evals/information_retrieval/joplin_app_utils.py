# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for Joplin app."""

import os
import random

from android_world.env import adb_utils
from android_world.env import interface
from android_world.task_evals.information_retrieval import proto_utils
from android_world.task_evals.information_retrieval.proto import state_pb2
from android_world.task_evals.information_retrieval.proto import task_pb2
from android_world.task_evals.utils import sqlite_schema_utils
from android_world.task_evals.utils import sqlite_utils
from android_world.utils import file_utils

_NOTES_TABLE = "notes"
_NOTES_NORMALIZED_TABLE = "notes_normalized"
_FOLDER_TABLE = "folders"
_DB_PATH = "/data/data/net.cozic.joplin/databases/joplin.sqlite"
_APP_NAME = "joplin"
# Sometimes this field gets added to the Joplin db, but we do not need it.
_EXCLUDE_FIELD = "deleted_time"


def setup_task_state(
    relevant_state: state_pb2.NotesApp,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
    env: interface.AsyncEnv,
) -> None:
  """Sets up the  state for the Joplin app.

  Args:
    relevant_state: The state to set up.
    exclusion_conditions: The exclusion conditions to use when generating random
      notes.
    env: The Android environment interface for database interaction.
  """
  clear_dbs(env)
  notes = []

  # Keep track of already created folders.
  folder_mapping = {}
  notes += _generate_random_notes(
      100,
      exclusion_conditions,
      [note.folder for note in relevant_state.notes],
      folder_mapping,
      env,
  )
  for note in relevant_state.notes:
    notes.append(_create_note_from_proto(note, folder_mapping, env))
  random.shuffle(notes)
  add_notes(notes, env)


def clear_dbs(env: interface.AsyncEnv) -> None:
  """Clears Joplin databases."""
  sqlite_utils.delete_all_rows_from_table(
      _FOLDER_TABLE, _DB_PATH, env, _APP_NAME
  )
  sqlite_utils.delete_all_rows_from_table(
      _NOTES_TABLE, _DB_PATH, env, _APP_NAME
  )
  sqlite_utils.delete_all_rows_from_table(
      _NOTES_NORMALIZED_TABLE, _DB_PATH, env, _APP_NAME
  )
  adb_utils.close_app(_APP_NAME, env.controller)  # Register changes.


def _get_folder_to_id(
    env: interface.AsyncEnv,
) -> dict[str, str]:
  """Gets a mapping from folder title to ID as represented in Folder table."""
  with env.controller.pull_file(_DB_PATH) as local_db_directory:
    local_db_path = file_utils.convert_to_posix_path(
        local_db_directory, os.path.split(_DB_PATH)[1]
    )
    folder_info = sqlite_utils.execute_query(
        f"select * from {_FOLDER_TABLE};",
        local_db_path,
        sqlite_schema_utils.JoplinFolder,
    )

  result = {}
  for row in folder_info:
    result[row.title] = row.id
  return result


def _add_folders(
    rows: list[sqlite_schema_utils.JoplinFolder],
    env: interface.AsyncEnv,
) -> None:
  """Inserts multiple folder rows into the remote Joplin database.

  Args:
      rows: A list of JoplinFolder instances to be inserted.
      env: The Android environment interface for database interaction.
  """

  sqlite_utils.insert_rows_to_remote_db(
      rows,
      _EXCLUDE_FIELD,
      _FOLDER_TABLE,
      _DB_PATH,
      _APP_NAME,
      env,
  )


def create_note(
    folder: str,
    title: str,
    body: str,
    folder_mapping: dict[str, str],
    env: interface.AsyncEnv,
    is_todo: int = False,
    todo_completed: bool = False,
) -> sqlite_schema_utils.JoplinNote:
  """Generates random note."""
  if not folder_mapping:
    folder_mapping.update(_get_folder_to_id(env))

  if folder not in folder_mapping:
    # Folder hasn't been created yet.
    _add_folders([sqlite_schema_utils.JoplinFolder(folder)], env)
    folder_mapping.clear()
    folder_mapping.update(_get_folder_to_id(env))
    if folder not in folder_mapping:
      raise ValueError("Something went wrong could not find or create folder.")
  parent_id = folder_mapping[folder]
  return sqlite_schema_utils.JoplinNote(
      parent_id=parent_id,
      title=title,
      body=body,
      is_todo=int(is_todo),
      todo_completed=int(todo_completed),
  )


def add_notes(
    rows: list[sqlite_schema_utils.JoplinNote],
    env: interface.AsyncEnv,
) -> None:
  """Inserts multiple note rows into the remote Joplin database."""
  sqlite_utils.insert_rows_to_remote_db(
      rows,
      None,
      _NOTES_TABLE,
      _DB_PATH,
      _APP_NAME,
      env,
  )
  sqlite_utils.insert_rows_to_remote_db(
      _normalize_notes(rows),
      None,
      _NOTES_NORMALIZED_TABLE,
      _DB_PATH,
      _APP_NAME,
      env,
  )


def _normalize_notes(
    notes: list[sqlite_schema_utils.JoplinNote],
) -> list[sqlite_schema_utils.JoplinNormalizedNote]:
  return [
      sqlite_schema_utils.JoplinNormalizedNote(
          id=note.id,
          parent_id=note.parent_id,
          title=note.title.lower(),
          body=note.body,
          is_todo=note.is_todo,
          todo_completed=note.todo_completed,
          user_created_time=note.user_created_time,
          user_updated_time=note.user_updated_time,
          latitude=note.latitude,
          longitude=note.longitude,
          altitude=note.altitude,
          source_url=note.source_url,
          todo_due=note.todo_due,
      )
      for note in notes
  ]


def list_notes(
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.JoplinNote]:
  return sqlite_utils.get_rows_from_remote_device(
      _NOTES_TABLE,
      _DB_PATH,
      sqlite_schema_utils.JoplinNote,
      env,
  )


def _create_note_from_proto(
    note: state_pb2.Note,
    folder_mapping: dict[str, str],
    env: interface.AsyncEnv,
) -> sqlite_schema_utils.JoplinNote:
  """Creates a JoplinNote object from a state_pb2.Note proto."""
  is_todo = note.is_todo.lower() == "true"
  todo_completed = note.todo_completed.lower() == "true"
  return create_note(
      note.folder,
      note.title,
      note.body,
      folder_mapping,
      env,
      is_todo,
      todo_completed,
  )


def _generate_random_notes(
    num_notes: int,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
    relevant_folders: list[str],
    folder_mapping: dict[str, str],
    env: interface.AsyncEnv,
) -> list[sqlite_schema_utils.JoplinNote]:
  """Generates random notes with the given exclusion conditions."""
  return sqlite_schema_utils.get_random_items(
      num_notes,
      generate_item_fn=lambda: _generate_random_note(
          relevant_folders, folder_mapping, env
      ),
      filter_fn=lambda x: _check_note_conditions(
          x, exclusion_conditions, folder_mapping
      ),
  )


def _generate_random_note(
    relevant_folders: list[str],
    folder_mapping: dict[str, str],
    env: interface.AsyncEnv,
):
  """Generates a single random sqlite_schema_utils.JoplinNote object."""
  new_note = state_pb2.Note()
  # add to relevant folders 30% of the time:
  add_to_relevant_folder = random.random() < 0.3
  if add_to_relevant_folder:
    folder = random.choice(relevant_folders)
    if folder not in _FOLDERS:
      raise ValueError("Unexpected folder name: {}".format(folder))
  else:
    folder = random.choice(list(_FOLDERS.keys()))
  new_note.folder = folder
  new_note.is_todo = str(random.choice([True, False]))
  if new_note.is_todo:
    new_note.todo_completed = random.choice(["True", "False"])
  random_note = random.choice(_FOLDERS[folder])

  new_note.title = random_note["title"]
  new_note.body = random_note["body"]
  note = _create_note_from_proto(new_note, folder_mapping, env)
  return note


def _check_note_conditions(
    note: sqlite_schema_utils.JoplinNote,
    exclusion_conditions: list[task_pb2.ExclusionCondition],
    folder_mapping: dict[str, str],
) -> bool:
  """Evaluates the specified task against a set of exclusion conditions.

  A note is considered eligible if it does not satisfy all of the conditions
  specified in the exclusion_conditions list. Each condition is checked against
  various fields of the note. The note is eligible if not all of these
  conditions are met, ensuring it doesn't fall under any exclusion criteria
  defined.

  Args:
    note: The note to check.
    exclusion_conditions: All the conditions the note will be checked against,
      if they are all met, this note should be excluded and does not meet the
      conditions.
    folder_mapping: A map from folder name to ID as represented in the Folder
      table.

  Returns:
    A bool, True if the note does not meet all of the exclusion conditions,
    False otherwise.
  """
  if not exclusion_conditions:
    return True
  # Keeps track of whether an exclusion condition is met.
  all_conditions_met = True
  for condition in exclusion_conditions:
    if condition.field == "title":
      all_conditions_met = all_conditions_met and proto_utils.compare(
          note.title.lower(),
          condition.operation,
          condition.value.lower(),
      )
    elif condition.field == "folder":
      folder_name = [
          key.lower()
          for (key, value) in folder_mapping.items()
          if note.parent_id == value
      ]
      all_conditions_met = all_conditions_met and proto_utils.compare(
          folder_name[0],
          condition.operation,
          condition.value.lower(),
      )
    elif condition.field == "is_todo":
      all_conditions_met = all_conditions_met and proto_utils.compare(
          note.is_todo,
          condition.operation,
          1 if condition.value.lower() == "true" else 0,
      )
    elif condition.field == "todo_completed":
      all_conditions_met = all_conditions_met and proto_utils.compare(
          note.todo_completed,
          condition.operation,
          1 if condition.value.lower() == "true" else 0,
      )

  return not all_conditions_met


_RECIPES = [
    {
        "title": "Zesty Quinoa Salad",
        "body": (
            "Ingredients:\nCooked quinoa, chopped cucumber, diced tomato,"
            " crumbled feta cheese, lemon vinaigrette\nInstructions:\nToss"
            " ingredients together. Season to taste."
        ),
    },
    {
        "title": "Peanut Butter Power Smoothie",
        "body": (
            "Ingredients:\nPeanut butter, banana, milk of choice, protein"
            " powder, ice\nInstructions:\nBlend until smooth and creamy."
        ),
    },
    {
        "title": "Cheesy Veggie Scramble",
        "body": (
            "Ingredients:\nEggs, shredded cheese, diced bell pepper, chopped"
            " spinach, hot sauce (optional)\nInstructions:\nSauté peppers and"
            " spinach. Whisk eggs with cheese, add to pan, and scramble. Top"
            " with hot sauce if desired."
        ),
    },
    {
        "title": "Tuna Salad Surprise",
        "body": (
            "Ingredients:\nCanned tuna, celery, mayonnaise, relish, crackers or"
            " bread\nInstructions:\nMix tuna, celery, mayonnaise, and relish."
            " Serve on crackers or bread."
        ),
    },
    {
        "title": "Spicy Black Bean Wrap",
        "body": (
            "Ingredients:\nBlack beans, salsa, shredded cheese, avocado,"
            " tortilla\nInstructions:\nWarm beans, top tortilla with beans,"
            " salsa, cheese, and avocado."
        ),
    },
    {
        "title": "Fruity Yogurt Parfait",
        "body": (
            "Ingredients:\nGreek yogurt, granola, mixed"
            " berries\nInstructions:\nLayer yogurt, granola, and berries in a"
            " glass or jar."
        ),
    },
    {
        "title": "Sweet Potato Hash",
        "body": (
            "Ingredients:\nDiced sweet potato, onion, breakfast sausage"
            " (optional), seasoning\nInstructions:\nCook sweet potatoes and"
            " onion until tender. Add sausage if desired. Season to taste."
        ),
    },
    {
        "title": "Hummus and Veggie Delight",
        "body": (
            "Ingredients:\nHummus, pita bread, cucumber slices, carrot"
            " sticks\nInstructions:\nSpread hummus on pita, top with cucumbers"
            " and carrots."
        ),
    },
    {
        "title": "Creamy Tomato Soup",
        "body": (
            "Ingredients:\nCanned tomatoes, heavy cream, basil, grilled cheese"
            " sandwich (for dipping)\nInstructions:\nBlend tomatoes and cream,"
            " heat gently. Season with basil. Serve with grilled cheese for"
            " dipping."
        ),
    },
    {
        "title": "Apple Cinnamon Overnight Oats",
        "body": (
            "Ingredients:\nRolled oats, milk of choice, grated apple, cinnamon,"
            " pinch of brown sugar\nInstructions:\nCombine oats, milk, apple,"
            " cinnamon, and brown sugar. Refrigerate overnight."
        ),
    },
    {
        "title": "Chicken Tikka Masala",
        "body": (
            "Marinated chicken cooked in a creamy tomato sauce with aromatic"
            " spices."
        ),
    },
    {
        "title": "Chocolate Chip Cookies",
        "body": (
            "Classic recipe for chewy cookies with chocolate chips and a hint"
            " of vanilla."
        ),
    },
    {
        "title": "Beef Stir-Fry",
        "body": (
            "Quick and easy stir-fry with tenderbeef, colorful vegetables, and"
            " a savory sauce."
        ),
    },
    {
        "title": "Vegetarian Chili",
        "body": (
            "Hearty chili packed with beans, vegetables, and spices, perfect"
            " for a cold day."
        ),
    },
    {
        "title": "Salmon with Roasted Vegetables",
        "body": (
            "Healthy and flavorful dish with baked salmon and seasonal"
            " vegetables."
        ),
    },
    {
        "title": "Homemade Pizza",
        "body": (
            "Pizza dough recipe, sauce options, topping ideas for a"
            " customizable pizza night."
        ),
    },
    {
        "title": "Pasta Carbonara",
        "body": (
            "Creamy pasta dish with pancetta, eggs, Parmesan cheese, and black"
            " pepper."
        ),
    },
    {
        "title": "Pad Thai",
        "body": (
            "Stir-fried rice noodles with tofu or shrimp, eggs, bean sprouts,"
            " and a tangy sauce."
        ),
    },
    {
        "title": "Chicken Pot Pie",
        "body": (
            "Comforting pie filled with chicken, vegetables, and creamy sauce,"
            " topped with flaky crust."
        ),
    },
    {
        "title": "Shrimp Scampi",
        "body": (
            "Garlic butter shrimp with pasta, lemon juice, white wine, and"
            " fresh herbs."
        ),
    },
    {
        "title": "French Onion Soup",
        "body": (
            "Rich and flavorful soup with caramelized onions, beef broth, and"
            " crusty bread topped with melted cheese."
        ),
    },
    {
        "title": "Vegetable Curry",
        "body": (
            "Aromatic curry with a variety of vegetables, coconut milk, and"
            " spices."
        ),
    },
    {
        "title": "Quinoa Salad",
        "body": (
            "Healthy and refreshing salad with quinoa, vegetables, herbs, and a"
            " lemon vinaigrette."
        ),
    },
    {
        "title": "Banana Bread",
        "body": (
            "Moist and flavorful bread made with ripe bananas, perfect for"
            " breakfast or a snack."
        ),
    },
    {
        "title": "Breakfast Burritos",
        "body": (
            "Scrambled eggs, sausage, cheese, and vegetables wrapped in a warm"
            " tortilla."
        ),
    },
    {
        "title": "Chocolate Mousse",
        "body": (
            "Decadent dessert made with chocolate, eggs, and cream, perfect for"
            " a special occasion."
        ),
    },
    {
        "title": "Apple Pie",
        "body": (
            "Classic American dessert with a flaky crust filled with sweet and"
            " tart apples."
        ),
    },
    {
        "title": "Brownies",
        "body": "Fudgy or cakey brownies with chocolate chips or nuts.",
    },
    {
        "title": "Pancakes",
        "body": (
            "Fluffy pancakes topped with butter, maple syrup, and fresh fruit."
        ),
    },
    {
        "title": "Smoothie Recipes",
        "body": (
            "Various combinations of fruits, vegetables, yogurt, and protein"
            " powder for healthy and refreshing smoothies."
        ),
    },
]

_TASKS = [
    {
        "title": "Morning Routine",
        "body": (
            "Tasks:\nMake bed\nShower and get dressed\nHealthy"
            " breakfast\nReview daily schedule"
        ),
    },
    {
        "title": "Website Updates",
        "body": (
            "Tasks:\nAdd new product photos\nUpdate contact form\nFix broken"
            " link on About page\nRun website speed test"
        ),
    },
    {
        "title": "Grocery Trip",
        "body": (
            "Tasks:\nCheck pantry staples\nMake a list of needed"
            " items\nRemember reusable bags\nCheck for coupons or deals"
        ),
    },
    {
        "title": "Travel Packing",
        "body": (
            "Tasks:\nCheck weather forecast\nChoose outfits and pack\nGather"
            " toiletries and essentials\nPrint travel documents"
        ),
    },
    {
        "title": "Apartment Cleanup",
        "body": (
            "Tasks:\nDo the dishes\nVacuum floors\nTidy living room\nTake out"
            " the trash"
        ),
    },
    {
        "title": "Project Brainstorm",
        "body": (
            "Tasks:\nDefine project goals\nFree-write potential ideas\nCreate a"
            " mind map\nIdentify next steps"
        ),
    },
    {
        "title": "Email Inbox Zero",
        "body": (
            "Tasks:\nDelete junk mail\nRespond to urgent emails\nOrganize"
            " important emails into folders\nUnsubscribe from unwanted lists"
        ),
    },
    {
        "title": "Workout Routine",
        "body": (
            "Tasks:\n5-minute warmup\n30 minutes cardio\nStrength"
            " training\nCool-down and stretching"
        ),
    },
    {
        "title": "Meal Planning",
        "body": (
            "Tasks:\nChoose recipes for the week\nMake a grocery list\nPrep"
            " ingredients if possible\nPlan for leftovers"
        ),
    },
    {
        "title": "Relax and Recharge",
        "body": (
            "Tasks:\nRead a book\nTake a relaxing bath\nListen to calming"
            " music\nGo for an evening walk"
        ),
    },
    {
        "title": "Grocery Shopping",
        "body": (
            "- Milk, eggs, bread \n- Fruits and vegetables \n- Chicken breast"
            " \n- Pasta \n- Toilet paper"
        ),
    },
    {
        "title": "Pay Bills",
        "body": (
            "- Electricity bill due May 15th \n- Internet bill due May 20th \n-"
            " Credit card payment due May 25th"
        ),
    },
    {
        "title": "Schedule Doctor's Appointment",
        "body": "Call Dr. Smith's office to schedule a check-up for next week.",
    },
    {
        "title": "Email Project Update to Client",
        "body": (
            "Send a summary of project progress and next steps to Acme Corp. by"
            " EOD."
        ),
    },
    {
        "title": "Finish Presentation Slides for Team Meeting",
        "body": "Complete slides on Q2 marketing strategy by Tuesday morning.",
    },
    {
        "title": "Book Flight for Summer Vacation",
        "body": "Research and book round-trip flights to Hawaii for July.",
    },
    {
        "title": "Renew Driver's License",
        "body": (
            "Visit the DMV to renew driver's license before it expires next"
            " month."
        ),
    },
    {
        "title": "Research Summer Camps for Kids",
        "body": (
            "Find options for summer camps that align with kids' interests and"
            " ages."
        ),
    },
    {
        "title": "Meal Prep for the Week",
        "body": (
            "Cook a large batch of chicken and vegetables for lunches and"
            " dinners."
        ),
    },
    {
        "title": "Clean Out Garage",
        "body": (
            "Sort through items, donate unwanted items, organize remaining"
            " items."
        ),
    },
    {
        "title": "Write Thank You Notes for Wedding Gifts",
        "body": "Send personalized thank you notes to all wedding guests.",
    },
    {
        "title": "Call Mom for Her Birthday",
        "body": "Wish Mom a happy birthday and catch up.",
    },
    {
        "title": "Schedule Oil Change for Car",
        "body": (
            "Make an appointment with the mechanic for an oil change and tire"
            " rotation."
        ),
    },
    {
        "title": "Research New Laptop",
        "body": "Compare features, prices, and reviews of different laptops.",
    },
    {
        "title": "Plant Vegetable Garden",
        "body": (
            "Buy seeds or seedlings, prepare soil, plant vegetables in raised"
            " beds."
        ),
    },
    {
        "title": "Organize Closet",
        "body": (
            "Declutter clothes, donate or sell unwanted items, rearrange"
            " remaining clothes."
        ),
    },
    {
        "title": "File Taxes",
        "body": (
            "Gather tax documents, complete tax return, submit online or by"
            " mail."
        ),
    },
    {
        "title": "Plan Weekend Getaway",
        "body": (
            "Research destinations, book accommodations, plan activities for a"
            " short trip."
        ),
    },
    {
        "title": "Learn New Skill",
        "body": (
            "Enroll in online course or workshop on photography, coding, or"
            " language learning."
        ),
    },
    {
        "title": "Set Up Retirement Account",
        "body": (
            "Open a Roth IRA or 401(k) and start contributing to retirement"
            " savings."
        ),
    },
]

_ATTENDEES = [
    "Emily",
    "John",
    "Sarah",
    "David",
    "Ava",
    "Michael",
    "Jessica",
    "Joshua",
]
_ACTION_ITEMS = [
    "Follow up with client on proposal",
    "Draft project timeline",
    "Research market trends",
    "Schedule team check-in",
    "Create design mockups",
    "Update website content",
    "Review budget report",
    "Send out meeting follow-up email",
    "Conduct user testing",
    "Finalize presentation materials",
    "Order supplies for event",
    "Coordinate with external vendors",
    "Submit reimbursement requests",
]

_MEETING_NOTES = [
    {
        "title": "Team Meeting - May 6, 2024",
        "body": (
            "Agenda, discussion points, action items, decisions made, next"
            " steps."
        ),
    },
    {
        "title": "Client Meeting - Acme Corp. - April 25, 2024",
        "body": (
            "Attendees, project updates, feedback, next steps, action items."
        ),
    },
    {
        "title": "Brainstorming Session - New Product Ideas - April 18, 2024",
        "body": (
            "Generated ideas, pros and cons, feasibility assessment, next"
            " steps."
        ),
    },
    {
        "title": "Project Kickoff Meeting - Website Redesign - April 10, 2024",
        "body": (
            "Project scope, timeline, team roles, communication plan, budget."
        ),
    },
    {
        "title": "One-on-One Meeting with John - April 3, 2024",
        "body": (
            "Performance feedback, career goals discussion, development"
            " opportunities."
        ),
    },
    {
        "title": "Board Meeting - Q1 Financial Results - March 28, 2024",
        "body": (
            "Financial report review, key performance indicators, budget"
            " discussion, future outlook."
        ),
    },
    {
        "title": "Weekly Team Update - March 21, 2024",
        "body": (
            "Progress updates on individual tasks, roadblocks, upcoming"
            " deadlines, team collaboration."
        ),
    },
    {
        "title": "Client Presentation - Proposal Review - March 14, 2024",
        "body": (
            "Proposal summary, client feedback, questions, revisions needed,"
            " next steps."
        ),
    },
    {
        "title": "Training Session - New Software - March 7, 2024",
        "body": (
            "Key features, how-to guide, troubleshooting tips, Q&A session."
        ),
    },
    {
        "title": "Conference Call - Remote Team - February 28, 2024",
        "body": (
            "Agenda, discussion points, action items for remote team"
            " collaboration and communication."
        ),
    },
    {
        "title": "Performance Review Meeting - Sarah - February 21, 2024",
        "body": (
            "Strengths, areas for improvement, goals for next quarter,"
            " development plan."
        ),
    },
    {
        "title": "Departmental Budget Meeting - February 14, 2024",
        "body": (
            "Budget review, cost-cutting measures, resource allocation,"
            " approval process."
        ),
    },
    {
        "title": "All-Hands Meeting - Company Update - February 7, 2024",
        "body": (
            "CEO presentation on company performance, new initiatives, Q&A"
            " session."
        ),
    },
    {
        "title": "Client Feedback Session - Project X - January 31, 2024",
        "body": (
            "Gathering feedback from client on project X, addressing concerns,"
            " identifying improvements."
        ),
    },
    {
        "title": "Strategic Planning Meeting - January 24, 2024",
        "body": (
            "Defining long-term goals, SWOT analysis, strategy development,"
            " implementation plan."
        ),
    },
    {
        "title": "Team Building Workshop - January 17, 2024",
        "body": (
            "Activities and exercises to improve communication, collaboration,"
            " and trust among team members."
        ),
    },
    {
        "title": "New Hire Orientation - January 10, 2024",
        "body": (
            "Welcome new employees, introduce company culture, provide"
            " onboarding information."
        ),
    },
    {
        "title": "Annual Performance Review - Self-Assessment - December 2023",
        "body": (
            "Reflect on accomplishments, challenges, areas for growth, goals"
            " for the coming year."
        ),
    },
    {
        "title": "Holiday Party Planning Meeting - December 2023",
        "body": (
            "Venue selection, catering options, entertainment, budget,"
            " decorations, guest list."
        ),
    },
    {
        "title": "Year-End Review Meeting - December 2023",
        "body": (
            "Summary of company performance, achievements, challenges, goals"
            " for the next year."
        ),
    },
    {
        "title": "Project Kickoff",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 3))
            + "\nAgenda:\nProject scope and objectives\nTimeline and"
            " milestones\nRoles and responsibilities\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 3))
        ),
    },
    {
        "title": "Marketing Strategy Brainstorm",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 2))
            + "\nAgenda:\nTarget audience analysis\nCampaign ideas\nBudget"
            " considerations\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 2))
        ),
    },
    {
        "title": "Website Redesign Review",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 4))
            + "\nAgenda:\nReview proposed wireframes\nDiscuss content"
            " updates\nFeedback on user experience\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 4))
        ),
    },
    {
        "title": "Quarterly Sales Meeting",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 3))
            + "\nAgenda:\nSales performance review\nNew product launch"
            " updates\nMarket analysis\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 2))
        ),
    },
    {
        "title": "Team Building Workshop",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 3))
            + "\nAgenda:\nTeam challenges discussion\nCommunication"
            " exercises\nGoal-setting activities\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 2))
        ),
    },
    {
        "title": "Client Project Update",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 3))
            + "\nAgenda:\nProject progress status\nChallenges and"
            " solutions\nBudget review\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 2))
        ),
    },
    {
        "title": "HR Policy Review",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 4))
            + "\nAgenda:\nReview updates to vacation policy\nDiscuss benefits"
            " package changes\nNew hire onboarding process\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 4))
        ),
    },
    {
        "title": "Design Sprint Planning",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 3))
            + "\nAgenda:\nDefine problem statement\nBrainstorm"
            " solutions\nPrototype and test ideas\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 2))
        ),
    },
    {
        "title": "Budget Review Meeting",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 2))
            + "\nAgenda:\nReview past quarter expenses\nAnalyze budget"
            " variances\nDiscuss upcoming project costs\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 2))
        ),
    },
    {
        "title": "All-Hands Team Meeting",
        "body": (
            "Attendees:\n"
            + "\n".join(random.sample(_ATTENDEES, 4))
            + "\nAgenda:\nCompany updates\nDepartment announcements\nCelebrate"
            " wins\nAction Items:\n"
            + "\n".join(random.sample(_ACTION_ITEMS, 3))
        ),
    },
]

_PERSONAL = [
    {
        "title": "Dream Journal Entry",
        "body": "Had a vivid dream about flyingover a vast ocean.",
    },
    {
        "title": "Bucket List",
        "body": "1. Learn to surf. 2. Visit Machu Picchu. 3. Write a novel.",
    },
    {
        "title": "Grocery List",
        "body": "Milk, eggs, bread, cheese, fruit, vegetables",
    },
    {
        "title": "Favorite Quotes",
        "body": (
            '"The only limit to our realization of tomorrow will be our doubts'
            ' of today." - Franklin D. Roosevelt'
        ),
    },
    {
        "title": "Movie Recommendations",
        "body": (
            "- Everything Everywhere All at Once \n- The Grand Budapest Hotel"
            " \n- Parasite"
        ),
    },
    {
        "title": "Birthday Gift Ideas for Mom",
        "body": "Spa day, gardening tools, personalized photo album",
    },
    {
        "title": "Workout Routine",
        "body": (
            "Monday: Cardio \nTuesday: Strength training \nWednesday: Rest"
            " \nThursday: Yoga \nFriday: Cardio \nWeekend: Active recovery"
        ),
    },
    {
        "title": "Travel Itinerary for Japan",
        "body": (
            "Day 1: Arrive in Tokyo, explore Shinjuku \nDay 2: Visit the"
            " Imperial Palace and Sensoji Temple \nDay 3: Take a day trip to"
            " Hakone \nDay 4: Travel to Kyoto, visit Kiyomizu-dera Temple \nDay"
            " 5: Explore Arashiyama Bamboo Forest \nDay 6: Depart from Osaka"
        ),
    },
    {
        "title": "Things I'm Grateful For",
        "body": "My family, my health, my friends, my home, my job",
    },
    {
        "title": "Home Improvement Projects",
        "body": (
            "Repaint the living room, install new kitchen backsplash, build a"
            " deck in the backyard"
        ),
    },
    {
        "title": "Party Planning Checklist",
        "body": (
            "Send invitations, plan menu, decorate venue, create playlist, hire"
            " photographer"
        ),
    },
    {
        "title": "Random Thoughts",
        "body": (
            "I wonder why cats purr? Is time travel possible? What's the"
            " meaning of life?"
        ),
    },
    {
        "title": "Password Ideas",
        "body": (
            "Combination of letters, numbers, symbols, not easily guessable"
        ),
    },
    {
        "title": "Favorite Recipes",
        "body": "Chocolate chip cookies, lasagna, chicken tikka masala",
    },
    {
        "title": "Book Recommendations",
        "body": (
            "- The Lord of the Rings \n- The Hitchhiker's Guide to the Galaxy"
            " \n- Pride and Prejudice"
        ),
    },
    {
        "title": "Song Lyrics I Love",
        "body": '"Imagine no possessions, I wonder if you can." - John Lennon',
    },
    {
        "title": "Things to Do This Weekend",
        "body": (
            "Hike in the mountains, visit a museum, have a picnic in the park"
        ),
    },
    {
        "title": "Self-Care Ideas",
        "body": (
            "Take a bubble bath, read a good book, meditate, spend time in"
            " nature"
        ),
    },
    {
        "title": "Personal Goals for the Year",
        "body": (
            "1. Save for a down payment on a house. 2. Get a promotion at work."
            " 3. Run a marathon."
        ),
    },
]

_WORK = [
    {
        "title": "Meeting Notes - Q2 Marketing Strategy",
        "body": (
            "Discussed social media campaigns, new product launch timeline,"
            " budget allocation."
        ),
    },
    {
        "title": "Project Timeline - Website Redesign",
        "body": (
            "Phase 1: Wireframes due May 15th \nPhase 2: Design approvals by"
            " June 1st \nPhase 3: Development complete by July 15th \nPhase 4:"
            " Launch by August 1st"
        ),
    },
    {
        "title": "Performance Review Talking Points",
        "body": (
            "- Exceeded sales targets by 15% \n- Successfully led"
            " cross-functional team \n- Developed new client onboarding process"
        ),
    },
    {
        "title": "Client Feedback - Acme Corp.",
        "body": (
            "Positive feedback on project delivery, requested additional"
            " features for Phase 2."
        ),
    },
    {
        "title": "To-Do List",
        "body": (
            "1. Respond to client emails \n2. Prepare presentation for team"
            " meeting \n3. Review budget proposal \n4. Schedule one-on-one with"
            " Sarah"
        ),
    },
    {
        "title": "Conference Notes - Tech Summit 2024",
        "body": (
            "Key takeaways on emerging technologies, potential applications for"
            " our industry."
        ),
    },
    {
        "title": "Team Brainstorming - New Product Ideas",
        "body": (
            "Generated 15 potential product ideas, will narrow down to top 3"
            " for further development."
        ),
    },
    {
        "title": "Employee Onboarding Checklist",
        "body": (
            "1. Set up workstation \n2. Provide access to company systems \n3."
            " Schedule training sessions \n4. Assign mentor"
        ),
    },
    {
        "title": "Company Policies and Procedures",
        "body": (
            "Links to documents on vacation policy, expense reimbursement, code"
            " of conduct."
        ),
    },
    {
        "title": "Travel Itinerary - Client Visit",
        "body": (
            "Flights booked, hotel reservations confirmed, meeting schedule"
            " finalized."
        ),
    },
    {
        "title": "KPI Report - Q1 2024",
        "body": (
            "Sales revenue up 10%, customer satisfaction rating at 92%,"
            " employee turnover rate at 5%."
        ),
    },
    {
        "title": "Code Snippets - Python",
        "body": (
            "Useful code examples for data analysis, web scraping, automation"
            " tasks."
        ),
    },
    {
        "title": "Industry News and Trends",
        "body": (
            "Summary of recent articles on market developments, competitor"
            " activity, regulatory changes."
        ),
    },
    {
        "title": "Job Descriptions - Open Positions",
        "body": (
            "Detailed descriptions for Marketing Manager, Software Engineer,"
            " Sales Representative roles."
        ),
    },
    {
        "title": "Meeting Minutes - Weekly Team Update",
        "body": (
            "Summary of discussion points, action items, decisions made during"
            " the meeting."
        ),
    },
    {
        "title": "Training Materials - New Software",
        "body": (
            "Step-by-step guides, video tutorials, FAQs for learning how to use"
            " the new software."
        ),
    },
    {
        "title": "Contact List - Key Clients",
        "body": (
            "Names, email addresses, phone numbers, company affiliations of"
            " important clients."
        ),
    },
    {
        "title": "Budget Proposal - 2025",
        "body": (
            "Detailed breakdown of projected expenses and revenue for each"
            " department."
        ),
    },
    {
        "title": "Professional Development Resources",
        "body": (
            "Links to online courses, workshops, conferences relevant to career"
            " growth."
        ),
    },
    {
        "title": "Team Building Activities",
        "body": (
            "Ideas for virtual and in-person activities to improve team morale"
            " and collaboration."
        ),
    },
]

_SCHOOL = [
    {
        "title": "Lecture Notes - Intro to Psychology",
        "body": (
            "Key concepts: nature vs. nurture, cognitive development, social"
            " psychology."
        ),
    },
    {
        "title": "Reading List - American Literature",
        "body": "- The Scarlet Letter \n- The Great Gatsby \n- Moby Dick",
    },
    {
        "title": "Study Guide - Calculus Midterm",
        "body": "Topics covered: derivatives, integrals, limits, applications.",
    },
    {
        "title": "Research Paper Outline - Climate Change",
        "body": (
            "I. Introduction \nII. Causes of Climate Change \nIII. Impacts on"
            " the Environment \nIV. Solutions \nV. Conclusion"
        ),
    },
    {
        "title": "Group Project Notes - Marketing Campaign",
        "body": (
            "Team members: Sarah, David, Emily. Due date: May 30th. Focus:"
            " promoting a new sustainable product."
        ),
    },
    {
        "title": "Exam Schedule - Spring Semester",
        "body": (
            "May 10th: Calculus \nMay 15th: American Literature \nMay 20th:"
            " Psychology"
        ),
    },
    {
        "title": "Class Syllabus - Introduction to Computer Science",
        "body": (
            "Course overview, grading policy, weekly schedule, required"
            " readings."
        ),
    },
    {
        "title": "Essay Draft - The Role of Technology in Education",
        "body": (
            "Discusses the benefits and challenges of integrating technology"
            " into classrooms."
        ),
    },
    {
        "title": "Lab Report - Chemistry Experiment",
        "body": "Purpose, materials, procedure, results, analysis, conclusion.",
    },
    {
        "title": "Flashcards - Spanish Vocabulary",
        "body": "Front: hola \nBack: hello",
    },
    {
        "title": "Scholarship Application Deadlines",
        "body": (
            "May 1st: National Merit Scholarship \nJune 1st: College Board"
            " Opportunity Scholarships"
        ),
    },
    {
        "title": "Student Club Meeting Notes - Debate Club",
        "body": (
            "Discussed upcoming tournament, new member recruitment, fundraising"
            " ideas."
        ),
    },
    {
        "title": "Campus Resources - Writing Center",
        "body": (
            "Offers one-on-one tutoring for essays, research papers, and other"
            " writing assignments."
        ),
    },
    {
        "title": "Professor Contact Information",
        "body": (
            "Dr. Smith: jsmith@university.edu \nDr. Johnson:"
            " sjohnson@university.edu"
        ),
    },
    {
        "title": "Financial Aid Checklist",
        "body": (
            "1. Submit FAFSA \n2. Apply for scholarships \n3. Contact financial"
            " aid office"
        ),
    },
    {
        "title": "Campus Event Calendar",
        "body": (
            "May 10th: Spring Concert \nMay 15th: Career Fair \nMay 20th: Guest"
            " Speaker Lecture"
        ),
    },
    {
        "title": "Study Tips for Final Exams",
        "body": (
            "Create a study schedule, review notes regularly, form study"
            " groups, practice with past exams."
        ),
    },
    {
        "title": "Internship Opportunities - Summer 2024",
        "body": (
            "Marketing internship at XYZ Company, Research internship at"
            " ABC Lab"
        ),
    },
    {
        "title": "Book Recommendations from Professor",
        "body": (
            "- Sapiens: A Brief History of Humankind \n- Thinking, Fast and"
            " Slow \n- Outliers: The Story of Success"
        ),
    },
    {
        "title": "Study Abroad Programs - Fall 2024",
        "body": (
            "Programs available in Spain, France, Italy, Germany, and Japan."
        ),
    },
]

_HOME = [
    {
        "title": "Home Maintenance Schedule",
        "body": (
            "Spring: clean gutters, check roof for damage, service AC \nSummer:"
            " mow lawn weekly, trim hedges, check sprinkler system \nFall: rake"
            " leaves, clean chimney, winterize pipes \nWinter: check for ice"
            " dams, shovel snow, change air filters"
        ),
    },
    {
        "title": "Grocery List",
        "body": (
            "- Milk \n- Eggs \n- Bread \n- Cheese \n- Fruits \n- Vegetables \n-"
            " Toilet paper"
        ),
    },
    {
        "title": "Recipe - Chicken Noodle Soup",
        "body": (
            "Ingredients: chicken, noodles, carrots, celery, onion, broth,"
            " herbs."
        ),
    },
    {
        "title": "Cleaning Checklist",
        "body": (
            "Kitchen: clean countertops, wipe down appliances, sweep and mop"
            " floor \nBathroom: clean toilet, sink, shower/tub, mirrors"
            " \nLiving room: dust furniture, vacuum carpet, fluff pillows"
        ),
    },
    {
        "title": "Home Renovation Ideas",
        "body": (
            "- Update kitchen cabinets \n- Refinish hardwood floors \n- Paint"
            " living room walls"
        ),
    },
    {
        "title": "Packing List - Summer Vacation",
        "body": (
            "- Clothes for warm weather \n- Swimsuit \n- Sunscreen \n- Hat \n-"
            " Sunglasses"
        ),
    },
    {
        "title": "Gardening Tips",
        "body": (
            "Water plants regularly, fertilize monthly, prune as needed, check"
            " for pests."
        ),
    },
    {
        "title": "Emergency Contact List",
        "body": "Police: 911 \nFire: 911 \nNeighbor: (123) 456-7890",
    },
    {
        "title": "Wi-Fi Password",
        "body": "Network Name: MyHomeWifi \nPassword: supersecretpassword",
    },
    {
        "title": "Home Inventory",
        "body": (
            "List of valuable items in case of insurance claim (electronics,"
            " jewelry, furniture)."
        ),
    },
    {
        "title": "Houseplant Care Guide",
        "body": (
            "Specific care instructions for each houseplant (watering"
            " frequency, light needs, soil type)."
        ),
    },
    {
        "title": "Utility Bill Due Dates",
        "body": (
            "Electricity: 15th of every month \nGas: 20th of every month"
            " \nWater: 5th of every month"
        ),
    },
    {
        "title": "Party Planning - Birthday",
        "body": "Guest list, menu, decorations, entertainment.",
    },
    {
        "title": "Neighborhood Watch Meeting Notes",
        "body": "Discussed recent crime trends, safety tips, upcoming events.",
    },
    {
        "title": "Pet Care Reminders",
        "body": (
            "Feed dog twice a day, walk dog daily, clean litter box, schedule"
            " vet checkups."
        ),
    },
    {
        "title": "DIY Project - Bookshelf",
        "body": "Materials needed: wood, screws, nails, saw, drill.",
    },
    {
        "title": "Movie Night Ideas",
        "body": "List of family-friendly movies to watch together.",
    },
    {
        "title": "Recipes to Try",
        "body": "Links or descriptions of new recipes to cook at home.",
    },
    {
        "title": "Home Security Checklist",
        "body": (
            "Lock doors and windows, install alarm system, set timers for"
            " lights, don't hide spare keys outside."
        ),
    },
    {
        "title": "Holiday Decoration Ideas",
        "body": "Themes, color schemes, DIY crafts, shopping list.",
    },
]

_PROJECTS = [
    {
        "title": "Community Garden Project",
        "body": (
            "Create a shared green space for the neighborhood, promoting"
            " sustainable food production and community connection."
        ),
    },
    {
        "title": "Home Renovation - Kitchen Remodel",
        "body": (
            "Design plans, budget, materials list, contractor quotes, timeline"
            " for a kitchen renovation."
        ),
    },
    {
        "title": "Mobile App Development - Expense Tracker",
        "body": (
            "Project outline, wireframes, technology stack, development"
            " timeline, marketing plan."
        ),
    },
    {
        "title": "Book Writing Project - Mystery Novel",
        "body": (
            "Outline, character sketches, plot points, research notes, writing"
            " schedule."
        ),
    },
    {
        "title": "Online Course Creation - Web Development Basics",
        "body": (
            "Course curriculum, lesson plans, video scripts, assessment"
            " questions, marketing strategy."
        ),
    },
    {
        "title": "DIY Furniture Building - Coffee Table",
        "body": (
            "Design plans, materials list, tools required, step-by-step"
            " instructions, finishing options."
        ),
    },
    {
        "title": "Photography Portfolio Website",
        "body": (
            "Website design mockups, image selection, content writing, hosting"
            " platform, launch plan."
        ),
    },
    {
        "title": "Charity Fundraising Event - 5K Run/Walk",
        "body": (
            "Event logistics, sponsorships, marketing plan, registration"
            " process, volunteer coordination."
        ),
    },
    {
        "title": "Small Business Launch - Handmade Jewelry",
        "body": (
            "Business plan, product line, branding, pricing, marketing"
            " strategy, online store setup."
        ),
    },
    {
        "title": "Art Installation - Public Sculpture",
        "body": (
            "Concept sketches, material selection, fabrication process,"
            " installation logistics, funding proposals."
        ),
    },
    {
        "title": "Documentary Film - Local Environmental Issues",
        "body": (
            "Research topics, interview subjects, filming locations, script"
            " outline, editing plan."
        ),
    },
    {
        "title": "Music Album Recording - Indie Rock Band",
        "body": (
            "Songwriting, studio booking, recording schedule, mixing and"
            " mastering, album artwork design."
        ),
    },
    {
        "title": "Community Theater Production - Shakespeare Play",
        "body": (
            "Casting calls, rehearsal schedule, set design, costume design,"
            " marketing plan."
        ),
    },
    {
        "title": "Coding Challenge - Machine Learning Algorithm",
        "body": (
            "Problem statement, data set, algorithm implementation, performance"
            " evaluation, results analysis."
        ),
    },
    {
        "title": "Website Redesign - Non-Profit Organization",
        "body": (
            "Needs analysis, wireframes, design mockups, content migration,"
            " development plan."
        ),
    },
    {
        "title": "Product Launch - Smart Home Device",
        "body": (
            "Market research, product specifications, pricing strategy,"
            " marketing campaign, launch timeline."
        ),
    },
    {
        "title": "Interior Design Project - Living Room Makeover",
        "body": (
            "Mood board, furniture selection, color palette, lighting plan,"
            " accessories."
        ),
    },
    {
        "title": "Travel Blog - Solo Trip Around Southeast Asia",
        "body": (
            "Itinerary, travel tips, destination highlights, photography plan,"
            " content schedule."
        ),
    },
    {
        "title": "Language Learning Project - Conversational Spanish",
        "body": (
            "Study plan, learning resources, practice activities, language"
            " exchange partners, progress tracking."
        ),
    },
    {
        "title": "Health and Fitness Challenge - 30-Day Transformation",
        "body": (
            "Workout plan, meal plan, progress tracking, motivation tips,"
            " before-and-after photos."
        ),
    },
]

_IDEAS = [
    {
        "title": "Personalized Pet Portraits",
        "body": (
            "Offer custom-painted portraits of pets based on photos provided by"
            " clients."
        ),
    },
    {
        "title": "Language Learning App",
        "body": (
            "Gamified language learning app with interactive exercises and"
            " personalized feedback."
        ),
    },
    {
        "title": "Sustainable Fashion Subscription Box",
        "body": (
            "Curated selection of eco-friendly clothing and accessories"
            " delivered monthly."
        ),
    },
    {
        "title": "Virtual Reality Escape Room",
        "body": "Immersive escape room experience using VR technology.",
    },
    {
        "title": "Food Delivery Service for Dietary Restrictions",
        "body": (
            "Cater to people with allergies, intolerances, or specific diets."
        ),
    },
    {
        "title": "Mental Health Support App",
        "body": (
            "Provides resources, guided meditations, and online therapy"
            " options."
        ),
    },
    {
        "title": "AI-Powered Personalized Travel Itinerary Generator",
        "body": (
            "Creates custom travel plans based on user preferences and"
            " interests."
        ),
    },
    {
        "title": "Smart Home Gardening System",
        "body": (
            "Automated watering, lighting, and nutrient monitoring for indoor"
            " plants."
        ),
    },
    {
        "title": "Subscription Box for Book Lovers",
        "body": (
            "Curated selection of books, bookish goodies, and exclusive author"
            " content."
        ),
    },
    {
        "title": "Online Platform for Local Artisans",
        "body": (
            "Showcase and sell handmade crafts and artwork directly to"
            " consumers."
        ),
    },
    {
        "title": "Eco-Friendly Cleaning Products",
        "body": (
            "Develop and market a line of sustainable cleaning products for"
            " households."
        ),
    },
    {
        "title": "Personalized Nutrition Coaching App",
        "body": (
            "Offers customized meal plans and fitness recommendations based on"
            " individual goals and needs."
        ),
    },
    {
        "title": "Social Media Platform for Pet Owners",
        "body": (
            "Connect with other pet owners, share photos, and find pet-related"
            " services."
        ),
    },
    {
        "title": "Online Marketplace for Vintage Clothing",
        "body": "Buy and sell unique vintage clothing and accessories.",
    },
    {
        "title": "Augmented Reality Furniture Shopping App",
        "body": (
            "Visualize how furniture would look in your home before buying."
        ),
    },
    {
        "title": "Subscription Service for Sustainable Home Goods",
        "body": (
            "Deliver eco-friendly household products and reusable alternatives"
            " to single-use items."
        ),
    },
    {
        "title": "Crowdfunding Platform for Creative Projects",
        "body": (
            "Support artists, musicians, filmmakers, and other creatives in"
            " funding their projects."
        ),
    },
    {
        "title": "Mobile App for Finding Local Volunteer Opportunities",
        "body": (
            "Connect volunteers with organizations in need of their skills and"
            " time."
        ),
    },
    {
        "title": "Online Marketplace for Personalized Gifts",
        "body": "Offer custom-made gifts for various occasions and interests.",
    },
    {
        "title": "Zero-Waste Grocery Store",
        "body": (
            "Sell bulk food items and package-free products to reduce waste."
        ),
    },
]

_HEALTH = [
    {
        "title": "Workout Routine - Strength Training",
        "body": (
            "Exercises for each muscle group, sets, reps, rest periods, weekly"
            " schedule."
        ),
    },
    {
        "title": "Meal Plan - Week of May 6th",
        "body": (
            "Breakfast, lunch, dinner, snacks for each day, grocery list,"
            " recipes."
        ),
    },
    {
        "title": "Doctor's Appointment Notes - May 3rd",
        "body": (
            "Summary of discussion with doctor, diagnosis, treatment plan,"
            " medication list, follow-up appointments."
        ),
    },
    {
        "title": "Medication Schedule",
        "body": (
            "List of medications, dosage, frequency, time to take, potential"
            " side effects, refills needed."
        ),
    },
    {
        "title": "Health Goals for 2024",
        "body": (
            "Lose 10 pounds, run a 5K, reduce stress, improve sleep quality,"
            " get regular checkups."
        ),
    },
    {
        "title": "Fitness Tracker Data - April 2024",
        "body": (
            "Steps taken, calories burned, active minutes, sleep duration,"
            " heart rate."
        ),
    },
    {
        "title": "Mental Health Resources",
        "body": (
            "Contact information for therapists, support groups, hotlines,"
            " websites, apps for mental well-being."
        ),
    },
    {
        "title": "Healthy Recipes to Try",
        "body": (
            "Links or descriptions of nutritious recipes for breakfast, lunch,"
            " dinner, snacks, desserts."
        ),
    },
    {
        "title": "Nutrition Tips",
        "body": (
            "Guidelines for balanced eating, portion control, healthy food"
            " swaps, meal prep ideas."
        ),
    },
    {
        "title": "Exercise Ideas",
        "body": (
            "Variety of workouts for different fitness levels and interests"
            " (cardio, strength, flexibility)."
        ),
    },
    {
        "title": "Sleep Hygiene Checklist",
        "body": (
            "Tips for creating a relaxing bedtime routine, improving sleep"
            " environment, getting quality sleep."
        ),
    },
    {
        "title": "Health Insurance Information",
        "body": (
            "Policy number, provider contact information, coverage details,"
            " copayments, deductibles."
        ),
    },
    {
        "title": "Allergy Information",
        "body": (
            "List of allergies, triggers, symptoms, treatment plan, emergency"
            " contact information."
        ),
    },
    {
        "title": "Medical History",
        "body": (
            "Summary of past illnesses, surgeries, medications, immunizations,"
            " family medical history."
        ),
    },
    {
        "title": "Weight Loss Progress Tracker",
        "body": (
            "Starting weight, current weight, goal weight, weight loss"
            " milestones, measurements."
        ),
    },
    {
        "title": "Meditation and Mindfulness Resources",
        "body": (
            "Guided meditations, mindfulness exercises, breathing techniques"
            " for stress reduction."
        ),
    },
    {
        "title": "Health-Related Articles and Blogs",
        "body": (
            "Links to informative articles on health topics, wellness trends,"
            " medical research."
        ),
    },
    {
        "title": "Health Challenges and Solutions",
        "body": (
            "Personal notes on overcoming health obstacles, strategies for"
            " managing chronic conditions."
        ),
    },
    {
        "title": "Fitness Class Schedule",
        "body": (
            "Days, times, locations of fitness classes (yoga, Pilates, Zumba,"
            " strength training)."
        ),
    },
    {
        "title": "Food Diary",
        "body": (
            "Record of daily food intake, calories, macronutrients,"
            " micronutrients, water intake."
        ),
    },
]

_TRAVEL = [
    {
        "title": "Trip Itinerary - Europe Summer 2024",
        "body": (
            "Flights, accommodations, transportation, daily activities,"
            " sightseeing plans, restaurant reservations."
        ),
    },
    {
        "title": "Packing List - Beach Vacation",
        "body": (
            "Clothing, toiletries, electronics, travel documents, beach gear,"
            " first-aid kit."
        ),
    },
    {
        "title": "Travel Budget - Southeast Asia Backpacking",
        "body": (
            "Estimated costs for flights, accommodation, food, transportation,"
            " activities, visas."
        ),
    },
    {
        "title": "Travel Insurance Information",
        "body": (
            "Policy number, provider contact information, coverage details,"
            " claim procedures."
        ),
    },
    {
        "title": "Language Phrasebook - Italian",
        "body": (
            "Common phrases for greetings, directions, ordering food, asking"
            " for help."
        ),
    },
    {
        "title": "Travel Tips - Staying Healthy Abroad",
        "body": (
            "Vaccinations, food and water safety, jet lag prevention, managing"
            " common illnesses."
        ),
    },
    {
        "title": "Bucket List Destinations",
        "body": (
            "Dream travel destinations with reasons for visiting and potential"
            " activities."
        ),
    },
    {
        "title": "Hotel Reviews - Paris",
        "body": (
            "Reviews of hotels in Paris based on location, amenities, price,"
            " service, cleanliness."
        ),
    },
    {
        "title": "Flight Confirmation - Round-trip to Tokyo",
        "body": (
            "Airline, flight numbers, departure and arrival times, seat"
            " assignments, baggage allowance."
        ),
    },
    {
        "title": "Restaurant Recommendations - Rome",
        "body": (
            "List of restaurants in Rome with cuisine type, price range,"
            " location, reviews."
        ),
    },
    {
        "title": "Travel Photography Tips",
        "body": (
            "Equipment recommendations, composition techniques, capturing"
            " different types of travel photos."
        ),
    },
    {
        "title": "Visa Requirements - China",
        "body": (
            "Information on visa types, application process, required"
            " documents, processing times."
        ),
    },
    {
        "title": "Travel Journal - Road Trip Across America",
        "body": (
            "Daily entries documenting experiences, observations, thoughts, and"
            " feelings during the trip."
        ),
    },
    {
        "title": "Transportation Options - London",
        "body": (
            "Information on public transportation (tube, buses), taxis,"
            " ride-sharing services, bike rentals."
        ),
    },
    {
        "title": "Travel Apps and Websites",
        "body": (
            "List of useful apps for booking flights, hotels, finding"
            " restaurants, translating languages, navigating."
        ),
    },
    {
        "title": "Cultural Etiquette Tips - Japan",
        "body": (
            "Customs and traditions to be aware of, do's and don'ts,"
            " appropriate behavior in different settings."
        ),
    },
    {
        "title": "Solo Travel Tips",
        "body": (
            "Advice on staying safe, meeting people, planning activities,"
            " budgeting for solo travelers."
        ),
    },
    {
        "title": "Travel Gear Checklist",
        "body": (
            "Essentials like luggage, backpacks, travel adapters, toiletries,"
            " first-aid kit, travel pillow."
        ),
    },
    {
        "title": "Festivals and Events Calendar - Europe",
        "body": (
            "List of upcoming festivals, cultural events, concerts, exhibitions"
            " in different European countries."
        ),
    },
    {
        "title": "Travel Photography Gear",
        "body": (
            "Camera, lenses, tripod, filters, memory cards, batteries, cleaning"
            " supplies."
        ),
    },
]

_FINANCE = [
    {
        "title": "Monthly Budget - May 2024",
        "body": (
            "Income, expenses, savings goals, spending categories, debt"
            " repayment plan."
        ),
    },
    {
        "title": "Investment Portfolio Summary",
        "body": (
            "Breakdown of investments (stocks, bonds, mutual funds),"
            " performance overview, asset allocation."
        ),
    },
    {
        "title": "Retirement Savings Plan",
        "body": (
            "Contribution schedule, target retirement age, projected retirement"
            " income, investment options."
        ),
    },
    {
        "title": "Tax Preparation Checklist - 2023",
        "body": (
            "Documents needed (W-2, 1099 forms), deductions to claim, tax"
            " filing deadline."
        ),
    },
    {
        "title": "Mortgage Payment Schedule",
        "body": (
            "Loan amount, interest rate, monthly payment, remaining balance,"
            " amortization schedule."
        ),
    },
    {
        "title": "Emergency Fund Progress",
        "body": (
            "Current balance, savings goal, monthly contributions, target"
            " amount (3-6 months of expenses)."
        ),
    },
    {
        "title": "Credit Card Statement - April 2024",
        "body": (
            "Transactions, due date, minimum payment, outstanding balance,"
            " rewards earned."
        ),
    },
    {
        "title": "Financial Goals for 2024",
        "body": (
            "Save for a down payment on a house, pay off student loan debt,"
            " increase retirement contributions."
        ),
    },
    {
        "title": "Investment Research - Tech Stocks",
        "body": (
            "Analysis of potential tech companies to invest in, growth"
            " projections, risk assessment."
        ),
    },
    {
        "title": "Budgeting Tips & Tricks",
        "body": (
            "Strategies for saving money, reducing expenses, tracking spending,"
            " automating savings."
        ),
    },
    {
        "title": "Financial Advisor Contact Information",
        "body": (
            "Name, email address, phone number, website of financial advisor."
        ),
    },
    {
        "title": "Online Banking Login Details",
        "body": (
            "Username, password, security questions, account numbers for online"
            " banking access."
        ),
    },
    {
        "title": "Insurance Policies Summary",
        "body": (
            "Coverage details for health, auto, home, life insurance policies,"
            " contact information for insurers."
        ),
    },
    {
        "title": "Debt Repayment Plan",
        "body": (
            "List of debts (credit cards, student loans), balances, interest"
            " rates, minimum payments, payoff strategies."
        ),
    },
    {
        "title": "Expense Tracking Spreadsheet",
        "body": (
            "Template for tracking daily expenses, categorizing spending,"
            " identifying areas for saving."
        ),
    },
    {
        "title": "Financial News & Analysis",
        "body": (
            "Summary of articles and reports on market trends, economic"
            " outlook, investment strategies."
        ),
    },
    {
        "title": "Personal Finance Resources",
        "body": (
            "Links to helpful websites, blogs, podcasts, books on personal"
            " finance topics."
        ),
    },
    {
        "title": "College Savings Plan - 529 Account",
        "body": (
            "Beneficiary information, investment options, contribution history,"
            " projected college costs."
        ),
    },
    {
        "title": "Estate Planning Documents",
        "body": (
            "Will, power of attorney, healthcare directive, beneficiaries,"
            " executor information."
        ),
    },
    {
        "title": "Charitable Giving Log",
        "body": (
            "Record of donations to charitable organizations, amounts, dates,"
            " tax-deductible status."
        ),
    },
]

# Folder names contains all possibilities for the revelant folder names
# in the task proto.
_FOLDERS = {
    "Recipes": _RECIPES,
    "Tasks": _TASKS,
    "Meeting Notes": _MEETING_NOTES,
    "Personal": _PERSONAL,
    "Work": _WORK,
    "School": _SCHOOL,
    "Home": _HOME,
    "Projects": _PROJECTS,
    "Ideas": _IDEAS,
    "Health": _HEALTH,
    "Travel": _TRAVEL,
    "Finance": _FINANCE,
}
