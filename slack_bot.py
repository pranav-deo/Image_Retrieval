def send_slack_notif(msg):
    import requests
    headers = {
        'Content-type': 'application/json',
    }

    data = '{"text":"' + msg + '"}'

    response = requests.post('https://hooks.slack.com/services/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', headers=headers, data=data)


# def send_slack_image(img):
#     import requests

#     headers = {
#         'Content-type': 'application/json',
#     }

#     files = {
#         'image': ('{}'.format(img), open('{}'.format(img), 'rb')),
#     }

#     response = requests.post('https://hooks.slack.com/services/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', headers=headers, files=files)


# if __name__ == '__main__':
#     send_slack_notif("trial")
