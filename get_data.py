""" Get data from database and first round process.

    NOTE: This file is meant to be run in the AWS EC2 instance
          to bypass the GFW
"""

import os
import re
import json
from collections import defaultdict
from mysql import connector
from mysql.connector.cursor import MySQLCursor
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

PATH = os.path.join(os.curdir, 'data')

def create_data_folder():
    """ Create folder for data stroage if it does not exist."""
    if os.path.exists(PATH) is False or os.path.isdir(PATH) is False:
        os.mkdir(PATH)

def fmt_date(dt, fmt='%Y-%m-%d'):
    """ Format the date object to string."""
    return dt.strftime(fmt)

def get_db_cnx():
    """ Connect to the databse if database is not connected
        Return a database cursor
    """
    db_config = {
        'host': os.getenv('DB_HOST'),
        'database': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USERNAME'),
        'password': os.getenv('DB_PASSWORD')
    }

    try:
        print('Connecting to Database mysql://{}/{}'.format(db_config['host'], db_config['database']))
        cnx = connector.connect(**db_config)
        print('Database connected')
    except connector.Error as err:
        if err.errno == connector.errorcode.ER_ACCESS_DENIED_ERROR:
            print('Username and password are not correct')
        if err.errno == connector.errorcode.ER_BAD_DB_ERROR:
            print('Requested database doesn\'t exist')
        print(err)
        exit(1)

    return cnx

def get_detailed_requirements(cursor):
    """ Select the data from database and strip out the text from html."""
    select_query =\
        """ SELECT projectId, challengeId, detailedRequirements, challengeTitle
            FROM Challenge
            WHERE projectId IN
            (
                SELECT t.projectId AS projectId FROM
                    (SELECT projectId, COUNT(*) AS numberOfChallenges
                    FROM Challenge
                    WHERE projectId != -1
                    GROUP BY projectId
                    ORDER BY projectId DESC) as t
                WHERE t.numberOfChallenges >= 10)
            ORDER BY projectId DESC;
        """
    cursor.execute(select_query)

    fetched_requirements = []

    for project_id, challenge_id, detailed_requirements, challenge_title in cursor:
        print(f'Selecting challenge {challenge_id}')
        fetched_requirements.append(
            {
                'project_id': project_id,
                'challenge_id': challenge_id,
                'requirements': detailed_requirements,
                'title': challenge_title
            }
        )

    print(f'Selected {len(fetched_requirements)} challenges.')
    with open(os.path.join(PATH, 'detail_requirements.json'), 'w') as fwrite:
        json.dump(fetched_requirements, fwrite, indent=4)

    cursor.close()

def get_number_of_challenges_by_project(cursor):
    """ Group the count of challenges by project id."""
    select_query =\
        """ SELECT
                projectId,
                MIN(DATE(registrationStartDate)) AS projectStartDate,
                COUNT(*) AS numOfChallenges,
                ROUND(SUM(totalPrize) / COUNT(*), 1) AS prizePerChallenge,
                CEIL(SUM(numberOfRegistrants) / COUNT(*)) AS registrantsPerChallenge
            FROM Challenge
            WHERE projectId != -1
            GROUP BY projectId
            ORDER BY projectId DESC;
        """
    cursor.execute(select_query)

    challenge_count_by_project = []
    challenge_info_by_project = []
    for project_id, project_start_date, number_of_challenges, prize_per_challenge, registrants_per_challenge in cursor:
        print(f'Project {project_id} | {number_of_challenges} challenges')
        challenge_count_by_project.append({'project_id': project_id, 'number_of_challenges': number_of_challenges})
        challenge_info_by_project.append({
            'project_id': project_id,
            'date': fmt_date(project_start_date),
            'number_of_challenges': number_of_challenges,
            'prize_per_challenge': float(prize_per_challenge),
            'registrants_per_challenge': int(registrants_per_challenge)
        })

    print(f'{len(challenge_count_by_project)} project in total.')
    with open(os.path.join(PATH, 'number_of_challenges_by_project.json'), 'w') as fwrite:
        json.dump(challenge_count_by_project, fwrite, indent=4)

    with open(os.path.join(PATH, 'challenge_info_by_project.json'), 'w') as fwrite:
        json.dump(challenge_info_by_project, fwrite, indent=4)

    cursor.close()

def get_tech_by_start_date(cursor):
    """ Get the technology string by start date."""
    select_query = """
        SELECT challengeId, technologies, Date(registrationStartDate) AS Dt
        FROM Challenge
        WHERE technologies != ''
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    tech_by_start_date = defaultdict(lambda: defaultdict(int))
    tech_by_challenge = []
    for challenge_id, tech_string, registration_start_date in cursor:
        print(f'Counting tech on {registration_start_date}')
        for tech_term in [tech_term.lower() for tech_term in tech_string.split(', ')]:
            print(f'\t{tech_term}')
            tech_by_start_date[fmt_date(registration_start_date)][tech_term] += 1

        tech_by_challenge.append({
            'challenge_id': challenge_id,
            'num_of_tech': len(tech_string.split(', ')),
            'tech_lst': tech_string.split(', '),
            'registration_start_date': fmt_date(registration_start_date)
        })

    with open(os.path.join(PATH, 'tech_by_start_date.json'), 'w') as fwrite:
        json.dump(tech_by_start_date, fwrite, indent=4)

    with open(os.path.join(PATH, 'tech_by_challenge.json'), 'w') as fwrite:
        json.dump(tech_by_challenge, fwrite, indent=4)

    cursor.close()

def get_total_prize_of_track_by_date(cursor):
    """ Get a table with track as column and date and index and total prize as value"""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(track = 'DEVELOP', totalPrize, 0)) AS Develop,
            SUM(IF(track = 'DESIGN', totalPrize, 0)) AS Design,
            SUM(IF(track = 'DATA_SCIENCE', totalPrize, 0)) AS Data_Science
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    prize_of_track_by_date = []
    for dt, develop, design, data_science in cursor:
        print(f'{dt} | dev {develop} | des {design} | ds {data_science}')
        prize_of_track_by_date.append(
            {
                'date': fmt_date(dt),
                'develop': round(float(develop), 2),
                'design': round(float(design), 2),
                'data_science': round(float(data_science), 2)
            }
        )

    print(f'{len(prize_of_track_by_date)} days of data retrieved')
    with open(os.path.join(PATH, 'prize_of_track_by_date.json'), 'w') as fwrite:
        json.dump(prize_of_track_by_date, fwrite, indent=4)

    cursor.close()

def get_number_of_track_by_date(cursor):
    """ Get the number of challenges aggregated by tracks and group by date"""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(track = 'DEVELOP', 1, 0)) AS Develop,
            SUM(IF(track = 'DESIGN', 1, 0)) AS Design,
            SUM(IF(track = 'DATA_SCIENCE', 1, 0)) AS Data_Science
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    number_of_track_by_date = []
    for dt, develop, design, data_science in cursor:
        print(f'{dt} | dev {develop} | des {design} | ds {data_science}')
        number_of_track_by_date.append({
            'date': fmt_date(dt),
            'develop': int(develop),
            'design': int(design),
            'data_science': int(data_science)
        })

    print(f'{len(number_of_track_by_date)} days of data retrieved')
    with open(os.path.join(PATH, 'number_of_track_by_date.json'), 'w') as fwrite:
        json.dump(number_of_track_by_date, fwrite, indent=4)

    cursor.close()

def get_total_prize_of_dev_subtrack_by_date(cursor):
    """ Get the total prize for sub tracks under DEVELOP track by date."""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(subTrack = 'ARCHITECTURE', totalPrize, 0)) AS ARCHITECTURE,
            SUM(IF(subTrack = 'ASSEMBLY_COMPETITION', totalPrize, 0)) AS ASSEMBLY_COMPETITION,
            SUM(IF(subTrack = 'BUG_HUNT', totalPrize, 0)) AS BUG_HUNT,
            SUM(IF(subTrack = 'CODE', totalPrize, 0)) AS CODE,
            SUM(IF(subTrack = 'CONCEPTUALIZATION', totalPrize, 0)) AS CONCEPTUALIZATION,
            SUM(IF(subTrack = 'CONTENT_CREATION', totalPrize, 0)) AS CONTENT_CREATION,
            SUM(IF(subTrack = 'COPILOT_POSTING', totalPrize, 0)) AS COPILOT_POSTING,
            SUM(IF(subTrack = 'DEVELOP_MARATHON_MATCH', totalPrize, 0)) AS DEVELOP_MARATHON_MATCH,
            SUM(IF(subTrack = 'FIRST_2_FINISH', totalPrize, 0)) AS FIRST_2_FINISH,
            SUM(IF(subTrack = 'SPECIFICATION', totalPrize, 0)) AS SPECIFICATION,
            SUM(IF(subTrack = 'TEST_SCENARIOS', totalPrize, 0)) AS TEST_SCENARIOS,
            SUM(IF(subTrack = 'TEST_SUITES', totalPrize, 0)) AS TEST_SUITES,
            SUM(IF(subTrack = 'UI_PROTOTYPE_COMPETITION', totalPrize, 0)) AS UI_PROTOTYPE_COMPETITION
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    prize_of_dev_subtrack_by_dt = []
    for dt, architecture, assembly, bug_hunt, code, conceptual, ctn_creation, copilot, marathon, f2f, spec, test_scenario, test_suite, ui_proto in cursor:
        prize_of_dev_subtrack_by_dt.append({
            'date': fmt_date(dt),
            'architecture': round(float(architecture), 2),
            'assembly_competition': round(float(assembly), 2),
            'bug_hunt': round(float(bug_hunt), 2),
            'code': round(float(code), 2),
            'conceptualization': round(float(conceptual), 2),
            'content_creation': round(float(ctn_creation), 2),
            'copilot_posting': round(float(copilot), 2),
            'dev_marathon': round(float(marathon), 2),
            'first_to_finish': round(float(f2f), 2),
            'specification': round(float(spec), 2),
            'test_scenario': round(float(test_scenario), 2),
            'test_suite': round(float(test_suite), 2),
            'ui_prototype': round(float(ui_proto), 2)
        })

    print(f'{len(prize_of_dev_subtrack_by_dt)} of dates data')
    with open(os.path.join(PATH, 'prize_of_dev_subtrack_by_dt.json'), 'w') as fwrite:
        json.dump(prize_of_dev_subtrack_by_dt, fwrite, indent=4)

    cursor.close()

def get_number_of_dev_subtrack_by_date(cursor):
    """ Get the number for sub tracks under DEVELOP track by date."""
    select_query = """
        SELECT
            DATE(registrationStartDate) as Dt,
            SUM(IF(subTrack = 'ARCHITECTURE', 1, 0)) AS ARCHITECTURE,
            SUM(IF(subTrack = 'ASSEMBLY_COMPETITION', 1, 0)) AS ASSEMBLY_COMPETITION,
            SUM(IF(subTrack = 'BUG_HUNT', 1, 0)) AS BUG_HUNT,
            SUM(IF(subTrack = 'CODE', 1, 0)) AS CODE,
            SUM(IF(subTrack = 'CONCEPTUALIZATION', 1, 0)) AS CONCEPTUALIZATION,
            SUM(IF(subTrack = 'CONTENT_CREATION', 1, 0)) AS CONTENT_CREATION,
            SUM(IF(subTrack = 'COPILOT_POSTING', 1, 0)) AS COPILOT_POSTING,
            SUM(IF(subTrack = 'DEVELOP_MARATHON_MATCH', 1, 0)) AS DEVELOP_MARATHON_MATCH,
            SUM(IF(subTrack = 'FIRST_2_FINISH', 1, 0)) AS FIRST_2_FINISH,
            SUM(IF(subTrack = 'SPECIFICATION', 1, 0)) AS SPECIFICATION,
            SUM(IF(subTrack = 'TEST_SCENARIOS', 1, 0)) AS TEST_SCENARIOS,
            SUM(IF(subTrack = 'TEST_SUITES', 1, 0)) AS TEST_SUITES,
            SUM(IF(subTrack = 'UI_PROTOTYPE_COMPETITION', 1, 0)) AS UI_PROTOTYPE_COMPETITION
        FROM Challenge
        GROUP BY Dt
        ORDER BY Dt ASC;
    """
    cursor.execute(select_query)

    number_of_dev_subtrack_by_dt = []
    for dt, architecture, assembly, bug_hunt, code, conceptual, ctn_creation, copilot, marathon, f2f, spec, test_scenario, test_suite, ui_proto in cursor:
        number_of_dev_subtrack_by_dt.append({
            'date': fmt_date(dt),
            'architecture': int(architecture),
            'assembly_competition': int(assembly),
            'bug_hunt': int(bug_hunt),
            'code': int(code),
            'conceptualization': int(conceptual),
            'content_creation': int(ctn_creation),
            'copilot_posting': int(copilot),
            'dev_marathon': int(marathon),
            'first_to_finish': int(f2f),
            'specification': int(spec),
            'test_scenario': int(test_scenario),
            'test_suite': int(test_suite),
            'ui_prototype': int(ui_proto)
        })

    print(f'{len(number_of_dev_subtrack_by_dt)} of dates data')
    with open(os.path.join(PATH, 'number_of_dev_subtrack_by_dt.json'), 'w') as fwrite:
        json.dump(number_of_dev_subtrack_by_dt, fwrite, indent=4)

    cursor.close()

def get_dev_track_info(cursor):
    """ Get more information of challenges under develop track"""
    select_query =\
        """ SELECT
                challengeId,
                DATE(registrationStartDate) AS Dt,
                totalPrize,
                numberOfRegistrants,
                subTrack
            FROM Challenge
            WHERE track = 'DEVELOP';
        """
    cursor.execute(select_query)

    dev_track_challenges = []
    for challenge_id, registration_start_date, total_prize, number_of_registrants, sub_track in cursor:
        print(f'Getting dev track info of {challenge_id}')
        dev_track_challenges.append({
            'challenge_id': challenge_id,
            'registration_start_date': fmt_date(registration_start_date),
            'total_prize': float(total_prize),
            'number_of_registrants': number_of_registrants,
            'sub_track': sub_track
        })

    with open(os.path.join(PATH, 'dev_track_challenges_info.json'), 'w') as fwrite:
        json.dump(dev_track_challenges, fwrite, indent=4)

    cursor.close()

def get_challenge_prz_and_avg_score(cursor):
    """ Retrieve challenge prize, avgerage score of all winners, number of submiiters, number of winners from database."""
    select_query =\
        """ SELECT
                C.challengeId AS challengeId,
                IFNULL(C.totalPrize, -1) AS totalPrize,
                IFNULL(W.avgScore, -1) AS avgScore,
                IFNULL(C.numberOfSubmitters, 0) AS numberOfSubmitters,
                IFNULL(W.numOfWinner, 0) AS numberOfWinner
            FROM (
                SELECT challengeId, projectId, totalPrize, numberOfSubmitters
                FROM Challenge
                WHERE projectId IN (
                    SELECT t.projectId AS projectId
                    FROM (
                        SELECT projectId, COUNT(*) AS numberOfChallenges
                        FROM Challenge
                        WHERE projectId != -1
                        GROUP BY projectId
                        ORDER BY projectId DESC
                        ) AS t
                    WHERE t.numberOfChallenges >= 10)
                ) AS C
            LEFT OUTER JOIN (
                SELECT
                    challengeId,
                    ROUND(AVG(points), 2) AS avgScore,
                    COUNT(*) AS numOfWinner
                FROM Challenge_Winner
                GROUP BY challengeId
                ) AS W
            ON C.challengeId = W.challengeId;
        """
    cursor.execute(select_query)

    challenge_prz_and_score = []
    for challenge_id, total_prize, avg_score, num_of_submitters, num_of_winners in cursor:
        print(f'Fetching {challenge_id} | prz {total_prize} | avg score {avg_score}')
        challenge_prz_and_score.append({
            'challenge_id': challenge_id,
            'total_prize': float(total_prize),
            'avg_score': float(avg_score),
            'num_of_submitters': int(num_of_submitters),
            'num_of_winners': int(num_of_winners)
        })

    print(f'Fetched {len(challenge_prz_and_score)} records.')

    with open(os.path.join(PATH, 'challenge_prz_and_score.json'), 'w') as fwrite:
        json.dump(challenge_prz_and_score, fwrite, indent=4)

    cursor.close()

def get_challenge_basic_info(cursor):
    """ Retrieve information like track, subtrack, registration
        start date, submission start date, submission end date
        of challenges
    """
    select_query_0 =\
        """ SELECT
                challengeId,
                projectId,
                totalPrize,
                track,
                subTrack,
                registrationStartDate,
                registrationEndDate,
                submissionEndDate,
                numberOfRegistrants,
                numberOfSubmissions,
                numberOfSubmitters
            FROM Challenge
            ORDER BY 
                FIELD(track, 'DEVELOP', 'DESIGN', 'DATA_SCIENCE'), 
                subTrack ASC;
    """
    cursor.execute(select_query_0)

    challenge_basic_info_dct = {}
    for challenge_id, project_id, total_prize, track, subtrack, reg_start_dt, reg_end_dt, sub_end_dt, num_of_reg, num_of_submission, num_of_submitters in cursor:
        print(f'Fetching basic info of challenge {challenge_id}')
        challenge_basic_info_dct[challenge_id] = {
            'challenge_id': challenge_id,
            'project_id': project_id,
            'total_prize': float(total_prize),
            'track': track,
            'subtrack': subtrack,
            'registration_start_date': fmt_date(reg_start_dt),
            'registration_end_date': fmt_date(reg_end_dt),
            'submission_end_date': fmt_date(sub_end_dt),
            'number_of_registration': int(num_of_reg),
            'number_of_submission': int(num_of_submission),
            'number_of_submitters': int(num_of_submitters)
        }

    select_query_1 = 'SELECT challengeId, platforms, technologies FROM Challenge;'
    cursor.execute(select_query_1)

    for challenge_id, platforms, technologies in cursor:
        challenge_basic_info_dct[challenge_id].update({
            'number_of_platforms': 0 if platforms == '' else len(platforms.split(', ')),
            'number_of_technologies': 0 if technologies == '' else len(technologies.split(', '))
        })

    challenge_basic_info = list(challenge_basic_info_dct.values())

    with open(os.path.join(PATH, 'challenge_basic_info.json'), 'w') as fwrite:
        json.dump(challenge_basic_info, fwrite, indent=4)

    cursor.close()

def get_challenge_registration_records(cursor):
    """ Retrieve ALLL challenge registration records."""
    select_query = 'SELECT * FROM Challenge_Registrant_Relation;'
    cursor.execute(select_query)

    cha_reg_records = defaultdict(list)
    for challenge_id, handle, registration_date, submission_date in cursor:
        print(f'Fethcing {challenge_id} - {handle}', end='\r')
        cha_reg_records[challenge_id].append({
            'challenge_id': challenge_id,
            'username': handle,
            'registration_date': fmt_date(registration_date),
            'submission_date': '' if submission_date is None else fmt_date(submission_date),
        })

    print('\nFetched {} challenges\' registrant records'.format(len(cha_reg_records)))

    for cha_id, cha_reg_info in cha_reg_records.items():
        print(f'Storing challenge registration {cha_id}.', end='\r')
        with open(os.path.join(PATH, 'challenge_registration', f'challenge_registration_{cha_id}.json'), 'w') as fwrite:
            json.dump(cha_reg_records, fwrite)

def main():
    """ Main entrance"""
    create_data_folder()
    cnx = get_db_cnx()
    # get_number_of_challenges_by_project(cnx.cursor())
    # get_total_prize_of_track_by_date(cnx.cursor())
    # get_number_of_track_by_date(cnx.cursor())
    # get_total_prize_of_dev_subtrack_by_date(cnx.cursor())
    # get_number_of_dev_subtrack_by_date(cnx.cursor())
    # get_tech_by_start_date(cnx.cursor())
    # get_dev_track_info(cnx.cursor())
    # get_challenge_prz_and_avg_score(cnx.cursor())
    # get_detailed_requirements(cnx.cursor())
    # get_challenge_basic_info(cnx.cursor())
    get_challenge_registration_records(cnx.cursor())
    cnx.close()

if __name__ == '__main__':
    main()
