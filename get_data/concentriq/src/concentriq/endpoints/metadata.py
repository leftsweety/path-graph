import json

from requests import Session


class Metadata:
    def __init__(self, concentriq_endpoint: str, session: Session):
        self.concentriq_endpoint = concentriq_endpoint
        self.metadata_fields_endpoint = f"{concentriq_endpoint}/metadata-fields"
        self.metadata_values_endpoint = f"{concentriq_endpoint}/metadata-values"
        self.session = session

    def create_metadata_field(self, field):
        return self.session.post(self.metadata_fields_endpoint, json=field, verify=False)

    def get_metadata_fields(self, filters=None):
        if filters is None:
            return self.session.get(self.metadata_fields_endpoint, verify=False)

        return self.session.get(self.metadata_fields_endpoint,
                                params={"filters": json.dumps(filters)}, verify=False)

    def get_fields_by_org_id(self, org_id):
        filters = {"organizationId": [org_id]}
        params = {"filters": json.dumps(filters)}
        return self.session.get(self.metadata_fields_endpoint, params=params, verify=False)

    def get_metadata_values(self, filters):
        return self.session.get(self.metadata_values_endpoint,
                                params={"filters": json.dumps(filters)}, verify=False)

    def update_metadata_values(self, metadata_values):
        return self.session.patch(self.metadata_values_endpoint,
                                  json={"metadataValues": metadata_values}, verify=False)

    def create_template(self, template):
        return self.session.post(f"{self.concentriq_endpoint}/templates",
                                 json=template, verify=False)

    def add_field_to_template(self, field_id, template_id):
        return self.session.post(f"{self.concentriq_endpoint}/template-fields",
                                 json={"fieldId": field_id, "templateId": template_id}, verify=False)
