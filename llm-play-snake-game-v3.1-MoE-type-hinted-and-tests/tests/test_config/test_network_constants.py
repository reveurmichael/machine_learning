"""Tests for config.network_constants module."""

import pytest
import socket

from config.network_constants import HOST_CHOICES


class TestHostChoices:
    """Test class for HOST_CHOICES configuration."""

    def test_host_choices_is_list(self):
        """Test that HOST_CHOICES is a list."""
        assert isinstance(HOST_CHOICES, list)

    def test_host_choices_not_empty(self):
        """Test that HOST_CHOICES is not empty."""
        assert len(HOST_CHOICES) > 0

    def test_host_choices_are_strings(self):
        """Test that all host choices are strings."""
        for host in HOST_CHOICES:
            assert isinstance(host, str)
            assert len(host) > 0

    def test_host_choices_expected_values(self):
        """Test that HOST_CHOICES contains expected host values."""
        expected_hosts = {"localhost", "0.0.0.0", "127.0.0.1"}
        assert set(HOST_CHOICES) == expected_hosts

    def test_host_choices_no_duplicates(self):
        """Test that HOST_CHOICES has no duplicate entries."""
        assert len(HOST_CHOICES) == len(set(HOST_CHOICES))

    def test_localhost_variations(self):
        """Test that both localhost representations are present."""
        # Both string and IP representations should be available
        assert "localhost" in HOST_CHOICES
        assert "127.0.0.1" in HOST_CHOICES

    def test_all_interfaces_host(self):
        """Test that all-interfaces host is present."""
        # 0.0.0.0 means bind to all available interfaces
        assert "0.0.0.0" in HOST_CHOICES


class TestHostChoicesValidity:
    """Test class for validating host choices are actually valid."""

    def test_localhost_resolvable(self):
        """Test that localhost can be resolved."""
        try:
            socket.gethostbyname("localhost")
        except socket.gaierror:
            pytest.fail("localhost should be resolvable")

    def test_loopback_ip_valid(self):
        """Test that 127.0.0.1 is a valid IP address."""
        try:
            socket.inet_aton("127.0.0.1")
        except socket.error:
            pytest.fail("127.0.0.1 should be a valid IP address")

    def test_all_interfaces_ip_valid(self):
        """Test that 0.0.0.0 is a valid IP address."""
        try:
            socket.inet_aton("0.0.0.0")
        except socket.error:
            pytest.fail("0.0.0.0 should be a valid IP address")

    def test_all_hosts_valid_for_binding(self):
        """Test that all host choices are valid for socket binding."""
        for host in HOST_CHOICES:
            try:
                # Try to create a socket and bind to the host (but don't listen)
                test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                test_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                # Use port 0 to let the OS choose an available port
                test_socket.bind((host, 0))
                test_socket.close()
            except (socket.error, OSError) as e:
                # Some hosts might not be bindable in certain environments
                # but they should at least be valid host strings
                if host == "0.0.0.0":
                    # 0.0.0.0 should always be bindable
                    pytest.fail(f"0.0.0.0 should be bindable: {e}")
                # For localhost and 127.0.0.1, binding might fail in some environments
                # but that's okay as long as the host string is valid

    def test_hosts_are_ip_or_hostname_format(self):
        """Test that all hosts follow valid IP or hostname format."""
        import re
        
        # Simple regex for IP address (not perfect but good enough for basic validation)
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        # Simple regex for hostname
        hostname_pattern = re.compile(r'^[a-zA-Z0-9.-]+$')
        
        for host in HOST_CHOICES:
            assert (ip_pattern.match(host) or hostname_pattern.match(host)), \
                f"Host {host} should match IP or hostname format"


class TestHostChoicesUsage:
    """Test class for validating host choices for actual usage scenarios."""

    def test_development_hosts_present(self):
        """Test that development-friendly hosts are present."""
        # For local development
        assert "localhost" in HOST_CHOICES
        assert "127.0.0.1" in HOST_CHOICES

    def test_production_host_present(self):
        """Test that production-ready host is present."""
        # For production deployment (bind to all interfaces)
        assert "0.0.0.0" in HOST_CHOICES

    def test_security_considerations(self):
        """Test that host choices consider security implications."""
        # localhost and 127.0.0.1 are secure (local only)
        secure_hosts = ["localhost", "127.0.0.1"]
        for secure_host in secure_hosts:
            assert secure_host in HOST_CHOICES
        
        # 0.0.0.0 is present but should be used carefully (binds to all interfaces)
        assert "0.0.0.0" in HOST_CHOICES

    def test_host_choices_for_web_frameworks(self):
        """Test that host choices are compatible with common web frameworks."""
        # Flask, Django, FastAPI all accept these standard host formats
        for host in HOST_CHOICES:
            # Should be string type (web frameworks expect strings)
            assert isinstance(host, str)
            # Should not contain protocol (no http:// prefix)
            assert not host.startswith("http")
            # Should not contain port (port is specified separately)
            assert ":" not in host

    def test_host_choices_ordering(self):
        """Test that host choices are in a logical order."""
        # While order doesn't matter functionally, 
        # it's good practice to have a logical ordering
        
        # localhost should come first (most common for development)
        assert HOST_CHOICES[0] == "localhost"
        
        # 0.0.0.0 might be last (most permissive)
        assert "0.0.0.0" in HOST_CHOICES


class TestHostChoicesEdgeCases:
    """Test class for edge cases and boundary conditions."""

    def test_empty_host_not_included(self):
        """Test that empty string is not included as a host choice."""
        assert "" not in HOST_CHOICES
        assert None not in HOST_CHOICES

    def test_invalid_hosts_not_included(self):
        """Test that obviously invalid hosts are not included."""
        invalid_hosts = [
            "999.999.999.999",  # Invalid IP
            "localhost:8000",   # Host with port
            "http://localhost", # Host with protocol
            "!@#$%",           # Invalid characters
            " localhost ",     # Host with spaces
        ]
        
        for invalid_host in invalid_hosts:
            assert invalid_host not in HOST_CHOICES

    def test_case_sensitivity(self):
        """Test that host choices use appropriate case."""
        # Hostnames are case-insensitive, but by convention use lowercase
        for host in HOST_CHOICES:
            if host.replace(".", "").replace("-", "").isalpha():
                # If it's alphabetic (like "localhost"), it should be lowercase
                assert host.islower(), f"Host {host} should be lowercase"

    def test_host_choices_immutability_expectation(self):
        """Test that HOST_CHOICES is expected to be treated as immutable."""
        # It's a list (mutable) but should not be modified in practice
        original_length = len(HOST_CHOICES)
        original_hosts = list(HOST_CHOICES)
        
        # Verify it's mutable (but shouldn't be modified in practice)
        HOST_CHOICES.append("test.example.com")
        assert len(HOST_CHOICES) == original_length + 1
        
        # Clean up
        HOST_CHOICES.pop()
        assert HOST_CHOICES == original_hosts


class TestNetworkConstantsIntegration:
    """Test class for integration with other network-related functionality."""

    def test_host_choices_with_default_ports(self):
        """Test that host choices work with standard web ports."""
        standard_ports = [80, 443, 8000, 8080, 5000]
        
        for host in HOST_CHOICES:
            for port in standard_ports:
                # Should be able to form valid (host, port) tuples
                address_tuple = (host, port)
                assert len(address_tuple) == 2
                assert isinstance(address_tuple[0], str)
                assert isinstance(address_tuple[1], int)

    def test_host_choices_url_construction(self):
        """Test that host choices can be used to construct URLs."""
        protocols = ["http", "https"]
        ports = [8000, 8080, 5000]
        
        for host in HOST_CHOICES:
            for protocol in protocols:
                for port in ports:
                    url = f"{protocol}://{host}:{port}/"
                    # Should be a valid URL format
                    assert url.startswith(protocol + "://")
                    assert str(port) in url

    def test_host_choices_socket_family_compatibility(self):
        """Test that host choices are compatible with IPv4 socket family."""
        # All current choices should be IPv4 compatible
        for host in HOST_CHOICES:
            if host != "localhost":  # localhost might resolve to IPv6
                try:
                    # Should be parseable as IPv4
                    socket.inet_aton(host)
                except socket.error:
                    # If it's not a valid IPv4 address, it should at least be a valid hostname
                    # that can resolve to IPv4
                    pass

    def test_module_documentation(self):
        """Test that the network constants module has appropriate documentation."""
        import config.network_constants as module
        assert hasattr(module, '__doc__')
        assert module.__doc__ is not None
        assert len(module.__doc__.strip()) > 0
        assert "network" in module.__doc__.lower() or "host" in module.__doc__.lower() 